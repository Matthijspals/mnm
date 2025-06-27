import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import orthogonal

from initialize_parameterize import *
from vi_rnn.transitions import Transition

class LRRNN(nn.Module):
    """
    Low-rank RNN
    Code inspired by https://github.com/DurstewitzLab/dendPLRNN
    """

    def __init__(self, dim_x, dim_z, dim_u, dim_N, dim_s, params):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space (rank)
            dim_u (int): dimensionality of the input
            dim_N (int): amount of neurons in the network
            dim_s (int): dimensionality of neuromodulator signals
            params (dict): dictionary of parameters
        """

        super(LRRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.d_u = dim_u
        self.d_N = dim_N
        self.d_s = dim_s 

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

        self.transition = Transition(
            self.d_x,
            self.d_z,
            self.d_u,
            self.d_N,
            self.d_s,
            nonlinearity=params["activation"],
            exp_par=params["exp_par"],
            shared_tau=params["shared_tau"],
            weight_dist=params["weight_dist"],
            m_orth=params["orth"],
            m_norm=params["m_norm"],
            weight_scaler=params["weight_scaler"],
            train_latent_bias=params["train_latent_bias"],
            train_neuron_bias=params["train_neuron_bias"],
            neuromodulation=None if "neuromodulation" not in params.keys() else params["neuromodulation"],
            train_nm_params=True if "train_nm_params" not in params.keys() else params["train_nm_params"]
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
            if self.transition.neuromodulation == 'additive': 
                self.get_initial_state = lambda u, s: self.initial_state.unsqueeze(
                    0
                ) + orth_proj(
                    self.transition.m_transform(self.transition.m),
                    torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
                ) 
                # + orth_proj(
                #     self.transition.m_transform(self.transition.m),
                #     (self.transition.A @ s.T).T
                # )
            else: 
                self.get_initial_state = lambda u, _: self.initial_state.unsqueeze(
                    0
                ) + orth_proj(
                    self.transition.m_transform(self.transition.m),
                    torch.einsum("Nu,Bu->BN", self.transition.Wu, u)
                )
        elif params["initial_state"] == "bias":
            self.get_initial_state = lambda u: -self.transition.h.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            ) 

    def forward(self, z, s=None, s_tilde=None, noise_scale=0, u=None, v=None, sim_v=False, sim_s=False):
        """forward step of the RNN, predict z one step ahead
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            noise_scale (float): scale of the noise
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
            s (torch.tensor; n_trials x dim_s x time_steps): neuromodulation

        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        if u is not None and sim_v==False:
            v = u
        if u is not None:
            v = self.transition.step_input(v, u)
        if s is not None and sim_s == False: 
            s_tilde = s 
        if s is not None: 
            s_tilde = self.transition.step_input(
                s_tilde.view(s_tilde.shape[0], s_tilde.shape[1], 1, 1),
                s.view(s.shape[0], s.shape[1], 1, 1))
            s_tilde = s_tilde.view(s_tilde.shape[0], s_tilde.shape[1])

        if noise_scale > 0:
            if self.params["scalar_noise_z"] == "Cov":
                cov_chol = self.chol_cov_embed(self.R_z)
                z = self.transition(z, v=v, s=s_tilde) + noise_scale * torch.einsum(
                    "xz, BzTK -> BxTK", cov_chol, self.normal.sample(z.shape)
                )
            else:
                z = self.transition(z, v=v, s=s_tilde) + noise_scale * self.normal.sample(
                    z.shape
                ) * self.std_embed_z(self.R_z).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            z = self.transition(z, v=v, s=s_tilde)
        return z, v, s_tilde

    def get_latent_time_series(
        self, time_steps=1000, cut_off=0, noise_scale=1, z0=None, u=None, s=None, sim_v=True, sim_s=False
    ):
        """
        Generate a latent time series of length time_steps
        Args:
            time_steps (int): length of the latent time series
            cut_off (int): cut off the first cut_off time steps
            noise_scale (float): scale of the noise
            z0 (torch.tensor; n_trials x dim_z x 1): initial latent state
            u (torch.tensor); n_trials x dim_u x time_steps): input
            s (torch.tensor); n_trials x dim_s x time_steps): neuromodulator states 
        Returns:
            Z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """ 
        with torch.no_grad():
            Z = []
            V = []
            S = []
            if z0 is None:
                z = torch.randn(1, self.d_z, 1, 1, device=self.R_x.device)
            else:
                if len(z0.shape)<=1 or z0.shape[0] == 1:  # only z dimension is given
                    z = z0.to(device=self.R_x.device).reshape(1, self.d_z, 1, 1)
                elif len(z0.shape) < 4:  # trial and z dimension is given
                    z = z0.to(device=self.R_x.device).reshape(
                        z0.shape[0], self.d_z, 1, 1
                    )
                else:
                    z = z0.to(device=self.R_x.device)
            #run model with input
            if u is not None:
                if len(u.shape) < 4:
                    u = u.unsqueeze(-1)  # add particle dim
                v = torch.zeros(u.shape[0], self.d_u, 1, 1,device=self.R_x.device)
                if s is not None: 
                    s_tilde = torch.zeros(s.shape[0], s.shape[1], device=self.R_x.device)
                for t in range(time_steps + cut_off):
                    
                    z,v,s_tilde = self.forward(
                            z, 
                            noise_scale=noise_scale, 
                            u=u[:, :, t].unsqueeze(2),
                            v=v,
                            s=None if s is None else s[:, :, t],
                            s_tilde=None if s is None else s_tilde,
                            sim_v=sim_v,
                            sim_s=sim_s
                        )
                    Z.append(z[:, :, 0])
                    V.append(v[:, :, 0])
                    S.append(s_tilde) 

                V = torch.stack(V)
                V = V[cut_off:]
                V = V.permute(1, 2, 0, 3)
               
            else:
                if s is not None: 
                    s_tilde = torch.zeros(s.shape[0], s.shape[1], device=self.R_x.device)
                else: 
                    s_tilde = None 

                for t in range(time_steps + cut_off):
                    z,_, s_tilde = self.forward(z, 
                                                s=s[:, :, t] if s is not None else None, 
                                                s_tilde=s_tilde, 
                                                sim_v=sim_v,
                                                sim_s=sim_s,
                                                noise_scale=noise_scale)
                    Z.append(z[:, :, 0])
                    S.append(s_tilde)

            # cut off the transients
            Z = torch.stack(Z)
            Z = Z[cut_off:]
            Z = Z.permute(1, 2, 0, 3)

            if s is not None: 
                S = torch.stack(S) 
                S = S[cut_off:]
                S = S.permute(1, 2, 0)

        if sim_v and sim_s:
            return Z, V, S 
        elif sim_v: 
            return Z, V
        elif sim_s: 
            return Z, S
        return Z

    def get_rates(self, z, u=None, s=None):
        """transform the latent states to the neuron activity"""
        R = self.transition.get_rates(z, u=u, s=s)
        return R

    def get_observation(self, z, v=None, s=None, noise_scale=0):
        """
        Generate observations from the latent states
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            noise_scale (float): scale of the noise
            s (torch.tensor; n_trials x dim_s x time_steps): Neuromodulation signals
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        if self.readout_rates == "rates":
            R = self.get_rates(z, u=v)
        elif self.readout_rates == "currents":
            m = self.transition.m_transform(self.transition.m)
            R = torch.einsum("Nz,BzTK->BNTK", m, z)
            if v is not None:
                Wu = self.transition.Wu
                R += torch.einsum("Nz,BzTK->BNTK", Wu, v)
            if s is not None and self.transition.neuromodulation == "additive":
                if len(s.shape) == 3:
                    s_x = torch.einsum("Ns,BsT->BNT", self.transition.A, s)
                else: 
                    s_x = (self.transition.A @ s.T).T
                    s_x = s_x.unsqueeze(-1)
                R += s_x.unsqueeze(-1)
                
        elif self.readout_rates == "z_and_v":
            R = torch.concat((z,v.repeat(1,1,1,z.shape[-1])),dim=1)
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
            m = self.transition.m_transform(self.transition.m)
            Wu = self.transition.Wu
            mW = torch.cat((m, Wu), dim=1)
            B_inv = torch.linalg.pinv(
                (
                    self.observation.cast_B(self.observation.B).T
                    @ mW
                ).T
            )[:,:self.d_z] 
        else:
            B_inv = torch.linalg.pinv(self.observation.cast_B(self.observation.B))

        if grad:
            return torch.einsum(
                "xz,bxT->bzT", (B_inv, X - self.observation.Bias.squeeze(-1))
            )
        else:
            return torch.einsum(
                "xz,bxT->bzT",
                (B_inv.detach(), X - self.observation.Bias.squeeze(-1).detach())
            )

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
            B = torch.zeros(self.dz, self.dx)
            B[range(self.dx), range(self.dx)] = 1
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

