import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import orthogonal

from initialize_parameterize import *
from vi_rnn.transitions.transitions import Transition
from vi_rnn.transitions.rank_scaling import RankScalingTransition
from vi_rnn.transitions.firing_rate_scaling import FiringRateScalingTransition
from vi_rnn.transitions.additive_input import AdditiveInputTransition

class LRRNN(nn.Module):
    """
    Low-rank RNN
    Code inspired by https://github.com/DurstewitzLab/dendPLRNN
    """

    def __init__(self, dim_x, dim_z, dim_u, dim_N, params):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space (rank)
            dim_u (int): dimensionality of the input
            dim_N (int): amount of neurons in the network
            params (dict): dictionary of parameters
        """

        super(LRRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.d_u = dim_u
        self.d_N = dim_N

        self.params = params
        self.normal = torch.distributions.Normal(0, 1)

        # Initialise noise
        # ------

        # need to keep diag positive
        self.chol_cov_embed = lambda x: torch.tril(x, diagonal=-1) + torch.diag_embed(
            torch.exp(x[range(x.shape[0]), range(x.shape[0])] / 2)
        )
        self.full_cov_embed = lambda x: self.chol_cov_embed(x) @ (
            self.chol_cov_embed(x).T
        )

        # initialise the observation noise
        # only used with Gaussian observations
        if params["scalar_noise_x"] == "Cov":
            self.R_x = nn.Parameter(
                torch.eye(self.d_x) * np.log(params["init_noise_x"]) * 2,
                requires_grad=params["train_noise_x"],
            )
            self.std_embed_x = lambda x: torch.sqrt(
                torch.diagonal(self.full_cov_embed(x))
            )
        elif params["scalar_noise_x"]:
            self.R_x = nn.Parameter(
                torch.ones(1) * np.log(params["init_noise_x"]) * 2,
                requires_grad=params["train_noise_x"],
            )
            self.std_embed_x = lambda log_var: torch.exp(log_var / 2).expand(self.d_x)
            self.var_embed_x = lambda log_var: torch.exp(log_var).expand(self.d_x)
        else:
            self.R_x = nn.Parameter(
                torch.ones(self.d_x) * np.log(params["init_noise_x"]) * 2,
                requires_grad=params["train_noise_x"],
            )
            self.std_embed_x = lambda log_var: torch.exp(log_var / 2)
            self.var_embed_x = lambda log_var: torch.exp(log_var)

        # initialise the latent noise
        if params["scalar_noise_z"] == "Cov":
            self.R_z = nn.Parameter(
                torch.eye(self.d_z) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z"],
            )
            self.std_embed_z = lambda x: torch.sqrt(
                torch.diagonal(self.full_cov_embed(x))
            )
        elif params["scalar_noise_z"]:
            self.R_z = nn.Parameter(
                torch.ones(1) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z"],
            )
            self.std_embed_z = lambda log_var: torch.exp(log_var / 2).expand(self.d_z)
            self.var_embed_z = lambda log_var: torch.exp(log_var).expand(self.d_z)
        else:
            self.R_z = nn.Parameter(
                torch.ones(self.d_z) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z"],
            )
            self.std_embed_z = lambda log_var: torch.exp(log_var / 2)
            self.var_embed_z = lambda log_var: torch.exp(log_var)

        #  initialise the latent noise for t = 0
        if params["scalar_noise_z_t0"] == "Cov":
            self.R_z_t0 = nn.Parameter(
                torch.eye(self.d_z) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z_t0"],
            )
            self.std_embed_z_t0 = lambda x: torch.sqrt(
                torch.diagonal(self.full_cov_embed(x))
            )
        elif params["scalar_noise_z_t0"]:
            self.R_z_t0 = nn.Parameter(
                torch.ones(1) * np.log(params["init_noise_z_t0"]) * 2,
                requires_grad=params["train_noise_z_t0"],
            )
            self.std_embed_z_t0 = lambda log_var: torch.exp(log_var / 2).expand(
                self.d_z
            )
            self.var_embed_z_t0 = lambda log_var: torch.exp(log_var).expand(self.d_z)
        else:
            self.R_z_t0 = nn.Parameter(
                torch.ones(self.d_z) * np.log(params["init_noise_z_t0"]) * 2,
                requires_grad=params["train_noise_z_t0"],
            )
            self.std_embed_z_t0 = lambda log_var: torch.exp(log_var / 2)
            self.var_embed_z_t0 = lambda log_var: torch.exp(log_var)

        # initialise the transition step
        # ---------
        if "clipped" in params.keys():
            if params["clipped"] and params["activation"] == "relu":
                params["activation"] = "clipped_relu"

        if params["neuromodulators"] and params["neuromodulation_type"] == "rank":
            print("Using rank-scaling neuromodulation")
            self.transition = RankScalingTransition(
                self.d_z,
                self.d_u,
                self.d_N,
                nonlinearity=params["activation"],
                exp_par=params["exp_par"],
                shared_tau=params["shared_tau"],
                weight_dist=params["weight_dist"],
                m_orth=params["orth"],
                m_norm=params["m_norm"],
                weight_scaler=params["weight_scaler"],
                train_latent_bias=params["train_latent_bias"],
                train_neuron_bias=params["train_neuron_bias"]
            )
        elif params["neuromodulators"] and params["neuromodulation_type"] == "firing_rate":
            print("Using firing rate scaling neuromodulation")
            self.transition = FiringRateScalingTransition(
                self.d_z,
                self.d_u,
                self.d_N,
                nonlinearity=params["activation"],
                exp_par=params["exp_par"],
                shared_tau=params["shared_tau"],
                weight_dist=params["weight_dist"],
                m_orth=params["orth"],
                m_norm=params["m_norm"],
                weight_scaler=params["weight_scaler"],
                train_latent_bias=params["train_latent_bias"],
                train_neuron_bias=params["train_neuron_bias"]
            )
        elif params["neuromodulators"] and params["neuromodulation_type"] == "additive":
            print("Using additive input neuromodulation")
            self.transition = AdditiveInputTransition(
                self.d_z,
                self.d_u,
                self.d_N,
                nonlinearity=params["activation"],
                exp_par=params["exp_par"],
                shared_tau=params["shared_tau"],
                weight_dist=params["weight_dist"],
                m_orth=params["orth"],
                m_norm=params["m_norm"],
                weight_scaler=params["weight_scaler"],
                train_latent_bias=params["train_latent_bias"],
                train_neuron_bias=params["train_neuron_bias"]
            )
        else:
            print("No neuromodulation applied")
            self.transition = Transition(
                self.d_z,
                self.d_u,
                self.d_N,
                nonlinearity=params["activation"],
                exp_par=params["exp_par"],
                shared_tau=params["shared_tau"],
                weight_dist=params["weight_dist"],
                m_orth=params["orth"],
                m_norm=params["m_norm"],
                weight_scaler=params["weight_scaler"],
                train_latent_bias=params["train_latent_bias"],
                train_neuron_bias=params["train_neuron_bias"],
            )

        # initialise the observation ste
        # ---------

        self.readout_rates = params["readout_rates"]

        # initialise the observation step, either readout from the latent states, or from the neuron activity
        if self.readout_rates == "rates":
            self.observation = Observation(
                self.d_z,
                self.d_x,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                identity_readout=params["identity_readout"],
                out_nonlinearity=params["out_nonlinearity"]

            )
        elif self.readout_rates == "currents":
            self.observation = Observation(
                self.d_N,
                self.d_x,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                identity_readout=params["identity_readout"],
                out_nonlinearity=params["out_nonlinearity"]

            )
        else:
            self.observation = Observation(
                self.d_z,
                self.d_x,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                identity_readout=params["identity_readout"],
                out_nonlinearity=params["out_nonlinearity"]
            )

        # initialise the initial state
        # ---------

        if params["initial_state"] == "zero":
            self.initial_state = nn.Parameter(
                torch.zeros(self.d_z), requires_grad=False
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )
        elif params["initial_state"] == "trainable":
            self.initial_state = nn.Parameter(torch.zeros(self.d_z), requires_grad=True)
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )
        elif params["initial_state"] == "bias":
            self.get_initial_state = lambda u: -self.transition.h.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

    def forward(self, z, s=None, noise_scale=0, u=None):
        """forward step of the RNN, predict z one step ahead
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            noise_scale (float): scale of the noise
            u (torch.tensor; n_trials x dim_u x time_steps x k): input

        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        if noise_scale > 0:
            if self.params["scalar_noise_z"] == "Cov":
                cov_chol = self.chol_cov_embed(self.R_z)
                z = self.transition(z, s, u=u) + noise_scale * torch.einsum(
                    "xz, BzTK -> BxTK", cov_chol, self.normal.sample(z.shape)
                )
            else:
                z = self.transition(z, s, u=u) + noise_scale * self.normal.sample(
                    z.shape
                ) * self.std_embed_z(self.R_z).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            z = self.transition(z, u=u, s=s)
        return z

    def get_latent_time_series(
        self, time_steps=1000, cut_off=0, noise_scale=1, z0=None, u=None, s=None
    ):
        """
        Generate a latent time series of length time_steps
        Args:
            time_steps (int): length of the latent time series
            cut_off (int): cut off the first cut_off time steps
            noise_scale (float): scale of the noise
            z0 (torch.tensor; n_trials x dim_z x 1): initial latent state
            u (torch.tensor); n_trials x dim_u x time_steps): input
            s (torch.tensor); n_trials x dim_z x time_steps): neuromodulator states 
        Returns:
            Z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        with torch.no_grad():
            Z = []
            if z0 is None:
                z = torch.randn(1, self.d_z, 1, 1, device=self.R_x.device)
            else:
                if len(z0.squeeze().shape) == 1:  # only z dimension is given
                    z = z0.to(device=self.R_x.device).reshape(1, self.d_z, 1, 1)
                elif len(z0.shape) < 4:  # trial and z dimension is given
                    z = z0.to(device=self.R_x.device).reshape(
                        z0.shape[0], self.d_z, 1, 1
                    )
                else:
                    z = z0.to(device=self.R_x.device)
            if u is not None and len(u.shape) < 4:
                u = u.unsqueeze(-1)  # add particle dim
            # if s is not None and len(s.shape) < 4: 
            #     s = s.unsqueeze(-1) # add particle dim

            for t in range(time_steps + cut_off):
                z = self.forward(
                    z, 
                    noise_scale=noise_scale, 
                    u=None if u is None else u[:, :, t].unsqueeze(2),
                    s=None if s is None else s[:, :, t]
                    # s=None if s is None else s[:, :, t].unsqueeze(2)
                )
                Z.append(z[:, :, 0])

            Z = torch.stack(Z)
            Z = Z[cut_off:]
            Z = Z.permute(1, 2, 0, 3)

        return Z

    def get_rates(self, z, u=None):
        """transform the latent states to the neuron activity"""
        R = self.transition.get_rates(z, u=u)
        return R

    def get_observation(self, z, noise_scale=0):
        """
        Generate observations from the latent states
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            noise_scale (float): scale of the noise
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        if self.readout_rates == "rates":
            R = self.get_rates(z)
        elif self.readout_rates == "currents":
            m = self.transition.m_transform(self.transition.m)
            R = torch.einsum("Nz,BzTK->BNTK", m, z)
        else:
            R = z
        X = self.observation(R)
        X += (
            noise_scale
            * self.normal.sample(X.shape)
            * self.std_embed_x(self.R_x).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        return X

    def inv_observation(self, X, grad=True):
        """
        Args:
            X (torch.tensor; n_trials x dim_x x time_steps): observations
            grad (bool): whether to allow autodiff through the inversion
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps): latent time series
        """
        if self.readout_rates == "currents":
            B_inv = torch.linalg.pinv(
                (
                    self.observation.cast_B(self.observation.B)
                    @ self.transition.m_transform(self.transition.m)
                ).T
            )

        else:
            B_inv = torch.linalg.pinv(self.observation.cast_B(self.observation.B))

        if grad:
            res = torch.einsum(
                "xz,bxT->bzT", (B_inv, X - self.observation.Bias.squeeze(-1))
            )
        else:
            res = torch.einsum(
                "xz,bxT->bzT",
                (B_inv.detach(), X - self.observation.Bias.squeeze(-1).detach()),
            )
        return res 

class Observation(nn.Module):
    """
    Readout from the latent states or the neuron activity
    """

    def __init__(
        self, dz, dx, train_bias=True, train_weights=True, identity_readout=False, out_nonlinearity='identity'
    ):
        """
        Args:
            dz (int): dimensionality of the latent space
            dx (int): dimensionality of the data
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            identity_readout (bool): whether to use the identity matrix as the readout matrix
        """
        super(Observation, self).__init__()
        self.dz = dz
        self.dx = dx

        if identity_readout:
            # B = torch.zeros(self.dx, self.dx)
            # B[range(self.dx), range(self.dx)] = 1
            B = torch.eye(self.dx)
            self.B = nn.Parameter(B, requires_grad=train_weights)
            self.mask = B
            self.cast_B = lambda x: x * self.mask
        else:
            self.B = nn.Parameter(
                np.sqrt(2 / dz) * torch.randn(self.dz, self.dx),
                requires_grad=train_weights,
            )
            self.cast_B = lambda x: x
            self.mask = torch.ones(1)

        self.Bias = nn.Parameter(
            torch.zeros(1, self.dx, 1, 1), requires_grad=train_bias
        )

        # for Poisson we need to rectify outputs to be positive
        if out_nonlinearity == "exp":
            self.nonlinearity = torch.exp
        elif out_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-10
        elif out_nonlinearity == "softplus":
            self.nonlinearity = torch.nn.functional.softplus
        elif out_nonlinearity == "identity":
            self.nonlinearity = lambda x: x

    def forward(self, z):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        return self.nonlinearity(torch.einsum("zx,bzTK->bxTK", (self.cast_B(self.B), z)) + self.Bias)

