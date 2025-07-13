# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F
import gpytorch
from gpytorch import kernels as gp_kernels
from gpytorch.distributions import MultitaskMultivariateNormal

import mbrl.util.math

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init
from mbrl.types import ModelInput
from mbrl.third_party.truncnorm.TruncatedNormal import TruncatedNormal


class BaseMultitaskGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, input_dim, num_models, num_inducing_points=100, use_coregionalization=True):
        # Initialize independent inducing points for each task/model
        inducing_points = torch.rand(1, num_models, num_inducing_points, input_dim)
        
        # Set the batch to learn a different variational distribution for each output dimension
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
           num_inducing_points, batch_shape=torch.Size([1, num_models])
        )
        
        # Wrap independent variational distributions together
        if use_coregionalization:
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=num_models,
                num_latents=num_models,
                latent_dim=-1,
            )
        else:
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=num_models
            )
        
        super(BaseMultitaskGPModel, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([1, num_models]))
        self.covar_module = gp_kernels.ScaleKernel( # learn the noise level in the target values
            gp_kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim, batch_shape=torch.Size([1, num_models])),
            batch_shape=torch.Size([1, num_models])
        )

    def forward(self, state):
        # Called from variational_strategy with [inducing_points, x] full input
        mean = self.mean_module(state)
        covar = self.covar_module(state)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class GPGaussianMLP(Ensemble):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        num_inducing_points: Optional[int] = 100,
        use_hot_gp: bool = False,
        use_thompson_sampling: bool = False,
        use_hucrl: bool = False,
        hucrl_beta: float = 1,
        hucrl_num_samples: int = 5,
        initial_low_reward_percentile: Optional[float] = None,
        final_low_reward_percentile: Optional[float] = None,
        high_reward_percentile: Optional[float] = None,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        if propagation_method is not None:
            raise RuntimeError("Propagation is not supported for GP MLP yet.")
    
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        self.in_size = in_size
        self.out_size = out_size

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        if deterministic:
            self.mean_and_logvar = create_linear_layer(hid_size, out_size)
        else:
            self.mean_and_logvar = create_linear_layer(hid_size, 2 * out_size)
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )
        
        self.gp = BaseMultitaskGPModel(in_size, out_size, num_inducing_points=num_inducing_points, use_coregionalization=True)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=out_size, batch_shape=torch.Size([1]))

        self.apply(truncated_normal_init)
        self.to(self.device)

        self.elite_models: List[int] = None

        # For optimism
        self.use_hot_gp = use_hot_gp
        self.use_thompson_sampling = use_thompson_sampling
        self.use_hucrl = use_hucrl
        self.hucrl_beta = hucrl_beta
        self.hucrl_num_samples = hucrl_num_samples
        self.initial_low_reward_percentile = initial_low_reward_percentile
        self.final_low_reward_percentile = final_low_reward_percentile
        self.low_reward_percentile = initial_low_reward_percentile
        self.high_reward_percentile = high_reward_percentile

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    def _default_forward( # for the mean function only
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            return mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., : self.out_size]
            logvar = mean_and_logvar[..., self.out_size :]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            return mean, logvar

    def _forward_ensemble( # for the mean function only
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is not None:
            raise RuntimeError("GP MLP does not support propagation yet.")
        mean, logvar = self._default_forward(x, only_elite=False)
        if self.num_members == 1:
            mean = mean[0]
            logvar = logvar[0] if logvar is not None else None
        return mean, logvar

    def forward( # for the mean function only
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)
    
    def predict_posterior(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        model_updates: Optional[int] = None,
    ):
        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Expand shape to match with inducing points
            x_expanded = x.unsqueeze(1).expand(-1, self.out_size, -1).unsqueeze(2)
            output = self.gp(x_expanded) # model posterior distribution for training data
            pred = self.likelihood(output) # posterior predicted distribution (probability distribution over the output value)
            nn_mean, _ = self._default_forward(x, only_elite=False)
            nn_mean = nn_mean.squeeze(0).unsqueeze(1)

        return MultitaskMultivariateNormal(pred.mean + nn_mean, pred.covariance_matrix) # add mean back
    
    def hucrl_sample(
        self,
        model_input,
        posterior,
    ):
        mean_reward = posterior.mean.squeeze(1)[:, -1]
        mean_obs = posterior.mean.squeeze(1)[:, :-1]
        stddev = posterior.stddev.squeeze(1)[:, :-1]
        all_rewards = [mean_reward]
        all_obs = [posterior.mean.squeeze(1)]
        for i in range(self.hucrl_num_samples):
            eta = torch.tensor(
                np.random.uniform(low=-1, high=1, size=(mean_obs.shape)),
                dtype=torch.float32,
            ).to(self.device)
            perturbation = torch.clamp(self.hucrl_beta * stddev * eta, min=-0.025, max=0.025)
            candidate_obs = mean_obs + perturbation
            candidate_model_input = torch.cat((candidate_obs, model_input[:, -(self.in_size - candidate_obs.shape[1]):]), dim=1)
            mean_output, _ = self._default_forward(candidate_model_input, only_elite=False)
            candidate_reward = mean_output.squeeze(0)[:, -1]
            all_rewards.append(candidate_reward)
            all_obs.append(torch.cat((candidate_obs, candidate_reward.unsqueeze(1)), dim=1))

        all_rewards = torch.stack(all_rewards, dim=0)
        all_obs = torch.stack(all_obs, dim=0)
        joint_indices = all_rewards.max(dim=0).indices[None,:,None].expand(all_obs.shape)
        max_obs = all_obs.gather(dim=0, index=joint_indices)[0]
        return max_obs
    
    def custom_sample_1d(
        self,
        model_input: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        model_updates: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Samples an output from the model using .

        This method will be used by :class:`ModelEnv` to simulate a transition of the form.
            outputs_t+1, s_t+1 = sample(model_input_t, s_t), where

            - model_input_t: observation and action at time t, concatenated across axis=1.
            - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
            - outputs_t+1: observation and reward at time t+1, concatenated across axis=1.

        The default implementation returns `s_t+1=s_t`.

        Args:
            model_input (tensor): the observation and action at.
            model_state (tensor): the model state st. Must contain a key
                "propagation_indices" to use for uncertainty propagation.
            deterministic (bool): if ``True``, the model returns a deterministic
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (`torch.Generator`, optional): an optional random number generator
                to use.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary. Everything but the observation is optional, and can
                be returned with value ``None``.
        """
        assert rng is not None
        posterior = self.predict_posterior(model_input, rng=rng, model_updates=model_updates)
        if self.use_hucrl:
            return self.hucrl_sample(model_input, posterior)
        elif not self.use_hot_gp:
            return posterior.sample().squeeze(1)
        
        # Sample next reward from the Truncated Normal distribution r' ~ p(r' | s, a, r' > k)
        reward_mean = posterior.mean[:, 0, -1]
        cov_matrix = posterior.covariance_matrix.detach().clone()
        cov_rr = cov_matrix[:, -1, -1]
        lower_threshold = torch.distributions.Normal(reward_mean, cov_rr).icdf(torch.tensor(self.low_reward_percentile))
        upper_threshold = torch.distributions.Normal(reward_mean, cov_rr).icdf(torch.tensor(self.high_reward_percentile))
        optimistic_reward_dist = TruncatedNormal(loc=reward_mean, scale=cov_rr, a=lower_threshold, b=upper_threshold)
        reward_sample = optimistic_reward_dist.sample()

        # Sample next state from the Multivariate Normal distribution s' ~ p(s' | s, a, r'=r')
        state_mean = posterior.mean[:, 0, :-1]
        cov_sr = cov_matrix[:, -1, :-1]
        cov_rs = cov_matrix[:, :-1, -1]
        cov_ss = cov_matrix[:, :-1, :-1]
        cov_sr_cov_rr_inv = cov_sr / (cov_rr.unsqueeze(-1))
        conditional_mean = state_mean + cov_sr_cov_rr_inv * (reward_sample - reward_mean).unsqueeze(-1)
        conditional_cov = cov_ss - cov_sr_cov_rr_inv.unsqueeze(-1) @ cov_rs.unsqueeze(1)
        conditional_dist = torch.distributions.MultivariateNormal(conditional_mean, conditional_cov)
        if self.use_thompson_sampling:
            state_sample = conditional_dist.sample()
        else:
            state_sample = conditional_dist.mean
        return torch.cat((state_sample, reward_sample.unsqueeze(1)), dim=1)
    
    def update_low_reward_percentile(self, total_model_updates, model_updates):
        self.low_reward_percentile = self.initial_low_reward_percentile - (self.initial_low_reward_percentile - self.final_low_reward_percentile) / total_model_updates * model_updates

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss for the mean function
        """
        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        else:
            return self._nll_loss(model_in, target), {}
        
    def set_gp_loss(self, dataset_train):
        self.gp_loss = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, num_data=dataset_train.num_stored)
        
    def gp_update(
        self,
        model_in: ModelInput,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        lambda_reg = 0
        optimizer.zero_grad()
        model_in = model_in.unsqueeze(1).expand(-1, self.out_size, -1).unsqueeze(2)
        target = target.unsqueeze(1)
        output = self.gp(model_in)
        elbo_loss = -self.gp_loss(output, target).sum()
        reg_loss = lambda_reg * torch.sum(output.covariance_matrix.diagonal(dim1=-2, dim2=-1))
        total_loss = elbo_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]
