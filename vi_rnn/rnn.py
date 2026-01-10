import torch
import torch.nn as nn
import numpy as np

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
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.dim_N = dim_N
        self.dim_s = dim_s 
        self.params = params
        self.normal = torch.distributions.Normal(0, 1)
        print(self.dim_N, self.dim_x)
        # Initialise noise
        # ------

   
        # Gaussian observations
        if params["obs_likelihood"] == "Gauss":
            self.observation_distribution = (
                lambda x, noise_scale=1: torch.distributions.Normal(
                    loc=x,
                    scale=self.std_embed_x(self.R_x).view(
                        1, self.dim_x, *([1] * len(x.shape[2:]))
                    )
                    * noise_scale,
                )
            )

        # Poisson observations
        elif params["obs_likelihood"] == "Poisson":
            self.observation_distribution = (
                lambda x, noise_scale=None: torch.distributions.Poisson(x)
            )

        else:
            raise ValueError(
                "observation_likelihood not recognised, use Gauss or Poisson"
            )
        


        if "noise_x" in params.keys():
            self.R_x, self.std_embed_x, self.var_embed_x = init_noise(
                params["noise_x"],
                self.dim_x,
                params["init_noise_x"],
                params["train_noise_x"],
            )
        if params["obs_likelihood"] == "Poisson" or params["obs_likelihood"] == "Gauss":
            # sampling and likelihood functions
            self.get_observation_log_likelihood = (
                lambda x_hat, x, noise_scale=1: self.observation_distribution(
                    x, noise_scale=noise_scale
                )
                .log_prob(x_hat)
                .sum(axis=1)
            )
            self.get_observation_sample = (
                lambda x, noise_scale=1: self.observation_distribution(
                    x, noise_scale
                ).sample()
            )
        self.obs_likelihood = params["obs_likelihood"]

        # Latent states transition noise
        self.R_z, self.std_embed_z, self.var_embed_z = init_noise(
            params["noise_z"],
            self.dim_z,
            params["init_noise_z"],
            params["train_noise_z"],
        )

        # Initial latent state noise
        self.R_z_t0, self.std_embed_z_t0, self.var_embed_z_t0 = init_noise(
            params["noise_z_t0"],
            self.dim_z,
            params["init_noise_z_t0"],
            params["train_noise_z_t0"],
        )
        # initialise the transition step
        # ---------
        if "clipped" in params.keys():
            if params["clipped"] and params["activation"] == "relu":
                params["activation"] = "clipped_relu"

        self.transition = Transition(
            self.dim_x,
            self.dim_z,
            self.dim_u,
            self.dim_N,
            self.dim_s,
            nonlinearity=params["activation"],
            decay=params["shared_tau"],
            weight_dist=params["weight_dist"],
            weight_scaler=params["weight_scaler"],
            train_latent_bias=params["train_latent_bias"],
            train_neuron_bias=params["train_neuron_bias"],
            neuromodulation=None if "neuromodulation" not in params.keys() else params["neuromodulation"],
            train_nm_params=True if "train_nm_params" not in params.keys() else params["train_nm_params"],
            train_decay=True if "train_alpha" not in params.keys() else params["train_alpha"],
        )

        # initialise the observation ste
        # ---------


        self.readout_from = params["readout_from"]

        # initialise the observation step, either readout from the latent states, or from the neuron activity
        if params["observation"] == "one_to_one":
            if self.readout_from == "rates":
                z_to_x_func = self.transition.get_rates
            elif self.readout_from == "currents":
                z_to_x_func = self.transition.get_currents
            else:
                raise ValueError(
                    "readout_from not recognised, use rates, currents (for a one_to_one obervation model)"
                )
            self.observation = One_to_One_observation(
                dim_x=self.dim_N,
                z_to_x_func=z_to_x_func,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                obs_nonlinearity=params["out_nonlinearity"],
            )
        elif params["observation"] == "affine":
            if self.readout_from == "z_and_v":
                dim_v = self.dim_u
            elif self.readout_from == "z":
                dim_v = 0
            else:
                raise ValueError(
                    "readout_from not recognised, use z_and_v, or z (for an affine observation model)"
                )
            self.observation = Affine_observation(
                dim_x=self.dim_x,
                dim_z=self.dim_z,
                dim_v=dim_v,
                train_bias=params["train_obs_bias"],
                train_weights=params["train_obs_weights"],
                obs_nonlinearity=params["obs_nonlinearity"],
            )


        else:
            raise ValueError(
                "observation not recognised, use one_to_one or affine,or calcium_one_to_one"
            )

        self.sim_v = params["sim_v"]
        self.sim_s = params["sim_s"]
        # initialise the initial state
        # ---------

        #TODO: Does this take into account neuromodulation and input correctly?

        if params["initial_state"] == "zero":
            self.initial_state = nn.Parameter(
                torch.zeros(self.dim_z), requires_grad=False
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )
        elif params["initial_state"] == "trainable":
            self.initial_state = nn.Parameter(torch.zeros(self.dim_z), requires_grad=True)
            if self.transition.neuromodulation == 'additive': 
                self.get_initial_state = lambda u, s: self.initial_state.unsqueeze(
                    0
                ) + orth_proj(
                    self.transition.m,
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
                    self.transition.m,
                    torch.einsum("Nu,Bu->BN", self.transition.Wu, u)
                )
        elif params["initial_state"] == "bias":
            self.get_initial_state = lambda u: -self.transition.h.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            ) 
    def get_latent_sample(self, z, noise_scale=0):
        """sample latent given mean at current timestep
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): mean at time t
            noise_scale (float): optional scale of the standard deviation
        Returns:
            z_sample (torch.tensor; n_trials x dim_z x time_steps x k): sample at time t
        """
        if self.params["noise_z"] == "full":
            cov_chol = chol_cov_embed(self.R_z)
            z_sample = z + noise_scale * torch.einsum(
                "xz, Bz... -> Bx...", cov_chol, self.normal.sample(z.shape)
            )
        else:
            z_sample = z + (
                noise_scale
                * self.normal.sample(z.shape)
                * self.std_embed_z(self.R_z).view(1, -1, *([1] * len(z.shape[2:])))
            )
        return z_sample

    def get_latent(self, z, v, s_tilde, noise_scale=0):
        """sample and mean given z at previous timestep
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): z at time t-1
            v (torch.tensor; n_trials x dim_u x time_steps x k): input
            noise_scale (float): optional scale of the standard deviation

        Returns:
            z_mean (torch.tensor; n_trials x dim_z x time_steps x k): mean at time t
            z_sample (torch.tensor; n_trials x dim_z x time_steps x k): sample at time t

        """
        z_mean = self.transition(z, v=v, s=s_tilde)
        z_sample = self.get_latent_sample(z_mean, noise_scale=noise_scale)

        return z_mean, z_sample


    def get_latent_time_series(
        self, time_steps=1000, cut_off=0, noise_scale=1, z0=None, u=None, s=None
    ):
        """
        Generate a latent time series of length time_steps
        Args:
            time_steps (int): length of the latent time series
            cut_off (int): cut off the first cut_off time steps
            noise_scale (float): scale of the noise
            z0 (torch.tensor; n_trials x dim_z x k): initial latent state
            u (torch.tensor); n_trials x dim_u x time_steps): input
            s (torch.tensor); n_trials x dim_s x time_steps): neuromodulator states 
        Returns:
            Z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """ 
        with torch.no_grad():
            Z = []
            V = []
            S = []

            if len(s.shape) < 4 and s is not None:
                s = s.unsqueeze(-1)  # add particle dimension
            if self.sim_s:
                s_tilde = torch.zeros(s.shape[0], s.shape[1], 1, device=self.R_x.device)
            else:
                s_tilde = s[:, :,0]  # initial neuromodulator state


            #run model with input
            if u is not None:
                if len(u.shape) < 4:
                    u = u.unsqueeze(-1)  # add particle dim
                if self.sim_v:    
                    v = torch.zeros(u.shape[0], self.dim_u, 1,device=self.R_x.device)
                else:
                    v = u[:, :, 0, :]  # initial input

                z =z0
                #Trials x dim_z x t x k
                Z.append(z0)
                V.append(v)
                S.append(s_tilde)
                for t in range(1,time_steps + cut_off):                   

                    _, z = self.get_latent(z, v, s_tilde, noise_scale=noise_scale)
                    
                    if self.sim_v:
                        v = self.transition.step_input(v, u[:, :, t - 1])
                    else:
                        v = u[:, :, t]

                    if self.sim_s:
                        s_tilde = self.transition.step_input(s_tilde, s[:, :, t-1])
                    else:
                        s_tilde = s[:, :, t]

                    Z.append(z)
                    V.append(v)
                    S.append(s_tilde) 
      

                V = torch.stack(V)
                V = V[cut_off:]
                V = V.permute(1, 2, 0, 3)
               
            else:
                Z.append(z0)
                S.append(s_tilde)
                for t in range(time_steps + cut_off):
                    _, z = self.get_latent(z, v, s_tilde, noise_scale=noise_scale)

                    if self.sim_s:
                        s_tilde = self.transition.step_input(s_tilde, s[:, :, t-1])
                    else:
                        s_tilde = s[:, :, t]

                    Z.append(z)
                    S.append(s_tilde)

            # cut off the transients
            Z = torch.stack(Z)
            Z = Z[cut_off:]
            Z = Z.permute(1, 2, 0, 3)

            if s is not None: 
                S = torch.stack(S) 
                S = S[cut_off:]
                S = S.permute(1, 2, 0, 3)

        return Z, V, S 




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
        X_mean = self.observation(z,v,s)
        X_sample = self.get_observation_sample(X_mean, noise_scale=noise_scale)

        return X_mean, X_sample


class One_to_One_observation(nn.Module):
    """
    Readout from the the activity of neurons in the network
    """

    def __init__(
        self,
        dim_x,
        z_to_x_func,
        train_bias=True,
        train_weights=True,
        obs_nonlinearity="identity",
    ):
        """
        Args:
            dim_x (int): dimensionality of the data
            z_to_x_func: maps latents to RNN unit space
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            obs_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(One_to_One_observation, self).__init__()
        self.dim_x = dim_x
        self.z_to_x_func = z_to_x_func
        self.B = nn.Parameter(
            torch.ones(self.dim_x),
            requires_grad=train_weights,
        )

        self.Bias = nn.Parameter(torch.zeros(self.dim_x), requires_grad=train_bias)

        # for Poisson we need to rectify outputs to be positive
        if obs_nonlinearity == "exp":
            exp = torch.exp 
            self.nonlinearity = lambda x: exp(x)
        elif obs_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) 
        elif obs_nonlinearity == "softplus":
            sp = torch.nn.functional.softplus
            self.nonlinearity = lambda x: sp(x)
        elif obs_nonlinearity == "identity":
            self.nonlinearity = lambda x:  x 
        else:
            raise ValueError(
                "obs_nonlinearity not recognised, use exp, relu, softplus, or identity"
            )

    def forward(self, z, v, s_tilde):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x k): observations
        """

        x = self.z_to_x_func(z, v,s_tilde)
        #print(np.min(x.detach().cpu().numpy()), np.max(x.detach().cpu().numpy()))
        x = x[:, :self.dim_x]
        bias = self.Bias.view(1, -1, *([1] * len(z.shape[2:])))
        B = self.B.view(1, -1, *([1] * len(z.shape[2:])))
        y = self.nonlinearity(B * x + bias) + 1e-6
        #print(np.min(y.detach().cpu().numpy()), np.max(y.detach().cpu().numpy()))

        return y



class Affine_observation(nn.Module):
    """
    Readout from the latent states
    """

    def __init__(
        self,
        dim_x,
        dim_z,
        dim_v=0,
        train_bias=True,
        train_weights=True,
        obs_nonlinearity="identity",
    ):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latents
            dim_v (int): dimensionality of the input
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            obs_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(Affine_observation, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_v = dim_v

        self.B = nn.Parameter(
            np.sqrt(2 / (self.dim_z + self.dim_v))
            * torch.randn(self.dim_z + self.dim_v, self.dim_x),
            requires_grad=train_weights,
        )

        self.Bias = nn.Parameter(torch.zeros(self.dim_x), requires_grad=train_bias)

        # for Poisson we need to rectify outputs to be positive
        if obs_nonlinearity == "exp":
            exp = torch.exp
            self.nonlinearity = lambda x: exp(x) + 1e-6
        elif obs_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-6
        elif obs_nonlinearity == "softplus":
            sp = torch.nn.functional.softplus
            self.nonlinearity = lambda x: sp(x) + 1e-6
        elif obs_nonlinearity == "identity":
            self.nonlinearity = lambda x: x + 1e-6
        else:
            raise ValueError(
                "obs_nonlinearity not recognised, use exp, relu, softplus, or identity"
            )

        # readout from z_and_v
        if self.dim_v > 0:
            self.cat_zv = lambda z, v: torch.concat(
                [(v.repeat(*([1] * len(v.shape[:-1])), z.shape[-1])), z], dim=1
            )
            """
            self.cat_zv = lambda z, v: torch.concat(
                [z, (v.repeat(*([1] * len(v.shape[:-1])), z.shape[-1]))], dim=1
            )
            """
        # or just z
        else:
            self.cat_zv = lambda z, v: z

    def forward(self, z, v):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        zv = self.cat_zv(z, v)
        bias = self.Bias.view(1, -1, *([1] * len(z.shape[2:])))
        return self.nonlinearity(torch.einsum("zx,bz...->bx...", (self.B, zv)) + bias)
