import torch
import torch.nn as nn

from initialize_parameterize import *
from vi_rnn.transitions.transitions import Transition

class RankScalingTransition(Transition):
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
        super(RankScalingTransition, self).__init__(dz, 
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
        R = self.get_rates(z, u=u)
        
        # Apply neuromodulator on N for all batches
        nm_n = s.unsqueeze(2) * self.n.unsqueeze(0)
        z = (
            A * z
            + torch.einsum("BzN,BNTK->BzTK", 
                        nm_n,
                        R)
            + self.hz.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
       
        return z