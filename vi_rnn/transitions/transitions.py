import torch 
import torch.nn as nn 
from torch.nn.utils.parametrizations import orthogonal
from initialize_parameterize import *

class Transition(nn.Module):
    """
    Latent dynamics of the prior
    """

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
        train_neuron_bias=True,
    ):
        """
        Args:
            dz (int): dimensionality of the latent space
            hidden_dim (int): amount of neurons in the network
            nonlinearity (str): nonlinearity of the hidden layer
            exp_par (bool): whether to use the exponential parameterisation for time constants
            shared_tau (bool): whether to have one shared time constant across the latent dimensions
            weight_dist (str): weight distribution
            m_orth (bool): whether to orthogonalise the left singular vectors
            m_norm (bool): whether to normalise the left singular vectors
            weight_scaler (float): scaling factor for the weights
            train_latent_bias (bool): whether to train the bias of the latents (z)
            train_neuron_bias (bool): whether to train the bias of the neurons (x)
        """
        super(Transition, self).__init__()
        self.dz = dz
        self.du = du

        # nonlinearity
        if nonlinearity == "relu":
            print("using ReLU activation")
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x - h)
            self.dnonlinearity = relu_derivative
        elif nonlinearity == "clipped_relu":
            print("using clipped ReLU activation")
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x + h) - relu(x)
            self.dnonlinearity = clipped_relu_derivative
        elif nonlinearity == "tanh":
            print("using tanh activation")
            self.nonlinearity = lambda x, h: torch.tanh(x - h)
            self.dnonlinearity = tanh_derivative
        elif nonlinearity == "identity":
            print("using identity activation")
            self.nonlinearity = lambda x, h: x - h
            self.dnonlinearity = lambda x: torch.ones_like(x)
        elif nonlinearity == "sigmoid": 
            print("using sigmoid activation")
            self.nonlinearity = lambda x, h: torch.sigmoid(x - h) 
            self.dnonlinearity = sigmoid_derivative  
        # time constants
        if shared_tau:
            if exp_par:
                self.AW = nn.Parameter(
                    torch.log(-torch.log(torch.ones(1, 1, 1, 1) * shared_tau))
                )
                self.cast_A = lambda x: torch.exp(-torch.exp(x))
            else:
                self.AW = nn.Parameter(torch.ones(1, 1, 1, 1) * shared_tau)
                self.cast_A = lambda x: x
        else:
            if exp_par:
                self.AW = init_AW_exp_par(self.dz)
                self.cast_A = exp_par_F
            else:
                self.AW = init_AW(self.dz)
                self.cast_A = lambda x: x.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # bias of the neurons
        if nonlinearity == "clipped_relu":
            self.h = nn.Parameter(
                uniform_init1d(hidden_dim), requires_grad=train_neuron_bias
            )
        else:
            self.h = nn.Parameter(
                torch.zeros(hidden_dim), requires_grad=train_neuron_bias
            )

        # bias of the latents
        self.hz = nn.Parameter(torch.zeros(dz), requires_grad=train_latent_bias)

        # weights (left and right singular vectors)
        if not m_orth:
            if weight_dist == "uniform":
                self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)
            elif weight_dist == "gauss":
                self.n, self.m = initialize_Ws_gauss(dz, hidden_dim)
            else:
                print("WARNING: weight distribution not implemented, using uniform")
                self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)
            self.m_transform = lambda x: x

        else:
            print("orthogonalising m")
            # Orthonormal columns
            self.m = orthogonal(nn.Linear(dz, hidden_dim, bias=False))
            self.n, _ = initialize_Ws_uniform(dz, hidden_dim)
            self.m_transform = lambda x: x.weight
        self.scaling = weight_scaler
        print("weight scaler", self.scaling)

        # Input weights
        if self.du > 0:
            self.Wu = nn.Parameter(
                uniform_init2d(hidden_dim, self.du), requires_grad=True
            )
        else:
            self.Wu = torch.zeros(hidden_dim, 0)

    def forward(self, z, s=None, u=None):
        """
        One step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        A = self.cast_A(self.AW)
        R = self.get_rates(z, u=u)
        z = (
            A * z
            + torch.einsum("zN,BNTK->BzTK", self.n * self.scaling, R)
            + self.hz.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        return z

    def get_rates(self, z, u=None):
        """Transform latents to neuron activity
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
        Returns:
            R (torch.tensor; n_trials x dim_N x time_steps x k): neuron activity"""
        if len(z.shape) == 3: z = z.unsqueeze(3) # add particle dimension 
        if u is not None and len(u.shape) == 3: u = u.unsqueeze(3)
        m = self.m_transform(self.m)
        X = torch.einsum("Nz,BzTK->BNTK", m, z)
        if u is not None:
            X += torch.einsum("Nu,BuTK->BNTK", self.Wu, u)
        R = self.nonlinearity(X, self.h.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        return R

    def jacobian(self, z):
        """Get jacobian along trajectory
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        Returns:
            jacobian (torch.tensor; n_trials x dim_z x dim_z x time_steps): jacobian along the trajectory
        """
        z = z.squeeze(-1)
        A = self.cast_A(self.AW).squeeze(-1)
        A = diag_mid(A)
        m = self.m_transform(self.m)
        X = torch.einsum("Nz,BzT->BNT", m, z)
        derivatives_act = self.dnonlinearity(X, self.h.unsqueeze(0).unsqueeze(2))
        proj_left = self.n.unsqueeze(0).unsqueeze(-1) * derivatives_act.unsqueeze(1)
        jacobian = A + torch.einsum("BzNT,Nx->BzxT", proj_left, m)
        return jacobian
