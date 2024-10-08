import torch
import torch.nn as nn

from initialize_parameterize import *
from vi_rnn.transitions.transitions import Transition

class FiringRateScalingTransition(Transition):
    def __init__(
        self,
        dz,
        du,
        hidden_dim,
        nonlinearity,
        exp_par,
        shared_tau,
        weight_dist="uniform",
        m_orth=False,
        m_norm=False,
        weight_scaler=1,
        train_latent_bias=True,
        train_neuron_bias=True
    ):
        super(FiringRateScalingTransition, self).__init__(dz, 
                         du, 
                         hidden_dim,
                         nonlinearity, 
                         exp_par, 
                         shared_tau, 
                         weight_dist, 
                         m_orth, 
                         m_norm,
                         weight_scaler, 
                         train_latent_bias, 
                         train_neuron_bias
                         )
         
        # initialize neuromodulator parameters from a standard normal distribution
        self.nm_params = nn.Parameter(torch.randn(self.dz, 1))

        

    def forward(self, z, s, u=None):
        """
        Perform the forward step
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            s (torch.tensor; n_trials x dim_z x time_steps): neuromodulator signals
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """

        A = self.cast_A(self.AW)
        # transform neuromodulator signals into a multiplicative constant
        neuromodulation = s @ self.nm_params
        R = self.get_rates(z, u=u, nm=neuromodulation.flatten())
  
        z = (
            A * z
            + torch.einsum("zN,BNTK->BzTK", self.n, R)
            + self.hz.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
       
        return z