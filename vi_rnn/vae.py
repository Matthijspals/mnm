import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from encoders import Inverse_Observation, CNN_encoder
from vi_rnn.rnn import LRRNN
import torch.nn as nn
import torch
import scipy
import numpy as np


class VAE(nn.Module):
    """
    VAE with low-rank RNN / dynamical systems prior
    """

    def __init__(self, vae_params):
        """initialize"""

        super(VAE, self).__init__()


        self.dim_x = vae_params["dim_x"]
        if "dim_x_hat" in vae_params:
            self.dim_x_hat = vae_params["dim_x_hat"]
        else:
            self.dim_x_hat = vae_params["dim_x"]

        if "dim_u" in vae_params:
            self.dim_u = vae_params["dim_u"]
        else:
            self.dim_u = 0

        self.dim_z = vae_params["dim_z"]
        self.dim_N = vae_params["dim_N"]
        self.dim_s = vae_params["dim_s"]
        self.vae_params = vae_params
        self.rnn = LRRNN(
            self.dim_x_hat,
            self.dim_z,
            self.dim_u,
            self.dim_N,
            self.dim_s,
            vae_params["rnn_params"],
        )
        self.has_encoder = True
        if vae_params["enc_architecture"] == "CNN":
            self.encoder = CNN_encoder(
                self.dim_x, self.dim_z, vae_params["enc_params"]
            )
        else:
            print("WARNING: no encoder")
            self.has_encoder = False

        self.min_var = 1e-6
        self.max_var = 100

        self.causal = vae_params["causal"]
        self.MSE_loss = nn.MSELoss()

   
    def to_device(self, device):
        """Move network between cpu / gpu (cuda)"""
        if self.has_encoder:
            self.encoder.to(device=device)
            self.encoder.normal.loc = self.encoder.normal.loc.to(device=device)
            self.encoder.normal.scale = self.encoder.normal.scale.to(device=device)
        self.rnn.to(device=device)
        self.rnn.normal.loc = self.rnn.normal.loc.to(device=device)
        self.rnn.normal.scale = self.rnn.normal.scale.to(device=device)
        self.rnn.transition.Wu = self.rnn.transition.Wu.to(device=device)

 