import torch 
import torch.nn as nn 
from initialize_parameterize import *

class Transition(nn.Module):
    """
    Latent dynamics of the prior
    """

    def __init__(
        self,
        dx,
        dz,
        du,
        hidden_dim,
        ds,
        nonlinearity,
        decay=0.9,
        weight_dist="uniform",
        weight_scaler=1,
        train_latent_bias=True,
        train_neuron_bias=True,
        neuromodulation=None,
        train_nm_params=True,
        train_decay=True,
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
            neuromodulation (str): neuromodulation type(s)
            train_nm_params (bool): whether to train neuromodulation parameters
        """
        super(Transition, self).__init__()
        self.dx = dx 
        self.dz = dz
        self.du = du
        self.ds = ds

        self.neuromodulation = neuromodulation
        #Do Glorot initialization
        if self.neuromodulation == "additive" or self.neuromodulation == "presynaptic" or self.neuromodulation == "postsynaptic":
            self.A = nn.Parameter(torch.empty(hidden_dim, self.ds), requires_grad=train_nm_params)
            self.A = nn.init.xavier_normal_(self.A)
        elif self.neuromodulation == "rank": 
            self.A = nn.Parameter(torch.empty(self.dz, self.ds), requires_grad=train_nm_params)
            self.A = nn.init.xavier_normal_(self.A)

        print(f'Neuromodulation type: {self.neuromodulation}')

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
        else:
            raise ValueError(
                "nonlinearity not recognised, use relu, clipped_relu, tanh, identity or sigmoid (logistic)"
            )
        self.decay_param = nn.Parameter(torch.log(-torch.log(torch.ones(1) * decay)), requires_grad=train_decay)

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
        if weight_dist == "uniform":
            self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)
        elif weight_dist == "gauss":
            self.n, self.m = initialize_Ws_gauss(dz, hidden_dim, weight_scaler)
        else:
            print("WARNING: weight distribution not implemented, using uniform")
            self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)

        self.scaling = weight_scaler
        print("weight scaler", self.scaling)
        # Input weights
        if self.du > 0:
            self.Wu = nn.Parameter(
                uniform_init2d(hidden_dim, self.du), requires_grad=True
            )
        else:
            self.Wu = torch.zeros(hidden_dim, 0)

    def forward(self, z, v=None, s=None):
        """
        One step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
            s (torch.tensor; n_trials x dim_s x time_steps x k): neuromodulation signal
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        # pre-synaptic and additive modify the currents 
        R = self.get_rates(z, s=s, v=v)
        n = self.n

        if self.neuromodulation == 'rank':
            s_z = torch.einsum("zs,Bs...->Bz...", self.A, s)
            z = (
                self.decay * z
                +  (torch.ones_like(s_z) + s_z) * torch.einsum("zN,BN...->Bz...", n * self.scaling, R)
                + self.hz.view(1, -1, *([1] * (len(z.shape)-2)))
            )
            return z
        
        
        elif self.neuromodulation == 'postsynaptic':
            s_x = torch.einsum("zs,Bs...->Bz...", self.A, s)
            R = (1 + s_x) * R
            z = (
                self.decay * z
                + torch.einsum("zN,BN...->Bz...", n * self.scaling , R)
                + self.hz.view(1, -1, *([1] * (len(z.shape)-2)))
            )
        return z
    
    @property
    def decay(self):
        return torch.exp(-torch.exp(self.decay_param)).view(1, 1, 1)

    def step_input(self, v, u):

        v= self.decay*v +(1-self.decay)*u
        return v
    
    def get_currents(self, z, v, s):
        """Transform latents to neuron activity, before nonlinearity
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input
        Returns:
            X (torch.tensor; n_trials x dim_N x time_steps x k): neuron activity before nonlinearity
        """
        #print(z.shape,v.shape,s.shape)
        X = torch.einsum("Nz,Bz...->BN...", self.m, z) + torch.einsum(
            "Nu,Bu...->BN...", self.Wu, v
        )

        if v is not None:
            X += torch.einsum("Nu,Bu...->BN...", self.Wu, v)

        if s is not None and self.neuromodulation != "rank":
            # transform neuromodulator signal to x space (i.e. from b x d_s -> b x d_x)
            #s_x =  (self.A @ s.T).T
            s_x  = torch.einsum("zs,Bs...->Bz...", self.A, s)

            if self.neuromodulation == 'additive':
                X += s_x

            elif self.neuromodulation == "presynaptic": 
                X = (1 + s_x) * X
        return X
    
    def get_rates(self, z, v=None, s=None):
        """Transform latents to neuron activity
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
            s (torch.tensor; n_trials x dim_s): neuromodulation
        Returns:
            R (torch.tensor; n_trials x dim_N x time_steps x k): neuron activity"""
        X = self.get_currents(z, v, s)
            
        R = self.nonlinearity(X, self.h.view(1, -1, *([1] * (len(X.shape)-2))))
        return R

