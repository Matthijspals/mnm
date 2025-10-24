import os 

import torch
import numpy as np
from torch.utils.data import Dataset

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from fixed_points.find_fixed_points_analytic import find_fixed_points_analytic

def np_relu(x):
    """ReLU function for numpy"""
    return np.maximum(x, 0)


def extract_phase_plane_rnn(rnn, xlims, ylims, n_points=30, inp=None):
    """Extract the phase plane of the RNN.
    Args:
        rnn: RNN, RNN model
        xlims: float or list, limits for x-axis
        ylims: float or list, limits for y-axis
        n_points: int, number of points to sample
        inp: np.array (n_inp), input to the RNN
    Returns:
        X: np.array (n_points x n_points), x-axis meshgrid
        Y: np.array (n_points x n_points), y-axis meshgrid
        u: np.array (n_points x n_points), x-axis velocity field
        v: np.array (n_points x n_points), y-axis velocity field
        norm: np.array (n_points x n_points), norm of the velocity field


    """
    U, V, B = extract_orth_basis_rnn(rnn)
    U = U.detach().numpy()
    V = V.detach().numpy()
    B = B.detach().numpy()
    if inp is None:
        I = np.zeros(1)
        inp = np.zeros(1)
    else:
        I = rnn.rnn.w_inp.detach().numpy()
    alpha = rnn.rnn.dt / rnn.rnn.tau

    if not isinstance(xlims, list):
        xlims = [-xlims, xlims]
    if not isinstance(ylims, list):
        ylims = [-ylims, ylims]

    def dyn_eq(x, y):
        z = np.array([x, y])
        dz = -z + V @ np_relu(U @ z + B + inp @ I) / alpha
        dz /= rnn.rnn.tau
        return dz[0], dz[1]

    X, Y = np.meshgrid(
        np.linspace(xlims[0], xlims[1], n_points),
        np.linspace(ylims[0], ylims[1], n_points),
    )
    u, v = np.zeros_like(X), np.zeros_like(X)
    NI, NJ = X.shape

    norm = np.zeros((NI, NJ))
    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            dx, dy = dyn_eq(x, y)
            u[i, j] = dx
            v[i, j] = dy
            norm[i, j] = np.log(np.linalg.norm([dx, dy]))
    return X, Y, u, v, norm


def extract_phase_plane_vae(vae, xlims, ylims, n_points=100, h=10, inp=None, nonlinearity='relu'):
    """Extract the phase plane of the vae.rnn.
    Args:
        vae: VAE, VAE model
        xlims: float, limits for x-axis
        ylims: float, limits for y-axis
        n_points: int, number of points to sample
        h: float, step size (in ms)
    Returns:
        X: np.array (n_points x n_points), x-axis meshgrid
        Y: np.array (n_points x n_points), y-axis meshgrid
        u: np.array (n_points x n_points), x-axis velocity field
        v: np.array (n_points x n_points), y-axis velocity field
        norm: np.array (n_points x n_points), norm of the velocity field"""
    prior = vae.rnn.transition
    decay = prior.cast_A(prior.AW).detach().numpy().squeeze()
    V = (prior.n * prior.scaling).detach().numpy()
    U = prior.m_transform(prior.m).detach().numpy()
    B = prior.h.detach().numpy()
    if inp is None:
        I = np.zeros(1)
        inp = np.zeros(1)
    else:
        I = prior.Wu.detach().numpy()

    if not isinstance(xlims, list):
        xlims = [-xlims, xlims]
    if not isinstance(ylims, list):
        ylims = [-ylims, ylims]

    def dyn_eq(x, y):
        z = np.array([x, y])
        if nonlinearity == 'relu':
            zn = (decay) * z + V @ np_relu(U @ z - B + I @ inp)
        elif nonlinearity == 'clipped_relu': 
            zn = (decay) * z + V @ np_relu(U @ z + B + I @ inp) - np_relu(U @ z + I @ inp)
            
        dz = (zn - z) / h
        return dz[0], dz[1]

    X, Y = np.meshgrid(
        np.linspace(xlims[0], xlims[1], n_points),
        np.linspace(ylims[0], ylims[1], n_points),
    )
    u, v = np.zeros_like(X), np.zeros_like(X)
    NI, NJ = X.shape

    norm = np.zeros((NI, NJ))
    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            dx, dy = dyn_eq(x, y)
            u[i, j] = dx
            v[i, j] = dy
            norm[i, j] = np.log(np.linalg.norm([dx, dy]))
    return X, Y, u, v, norm


def orthogonalise_network(vae):
    """
    Orthogonalise loadings of the LR RNN prior

    Warning: at the moment this makes the network unable
    to train afterwards as the cholesky decomposition is not constrained

    Additionally the output mapping might need to be adjusted

    Args:
        vae (VAE): VAE model
    Returns:
        vae (VAE): VAE model with orthogonalised loadings

    """
    with torch.no_grad():
        m_or = vae.rnn.transition.m  # 20,2
        n_or = vae.rnn.transition.n
        J = m_or @ n_or
        u, s, v = torch.linalg.svd(J)
        projection_matrix = u[:, : vae.dim_z].T @ m_or

        if vae.rnn.params["scalar_noise_z"] == "Cov":
            proj_chol = vae.rnn.chol_cov_embed(vae.rnn.R_z)
        else:
            proj_chol = torch.diag(vae.rnn.std_embed_z(vae.rnn.R_z))

        proj_chol = projection_matrix @ proj_chol
        m_new = u[:, : vae.dim_z]
        n_new = (v[: vae.dim_z].T * s[: vae.dim_z]).T
        vae.rnn.transition.m.copy_(m_new)
        vae.rnn.transition.n.copy_(n_new)
        vae.rnn.chol_cov_embed = lambda x: torch.tril(x)
        vae.rnn.R_z = torch.nn.Parameter(torch.linalg.cholesky(proj_chol @ proj_chol.T))
        vae.rnn.params["scalar_noise_z"] = "Cov"
    return vae


def extract_orth_basis_rnn(rnn):
    """
    Extract orthogonal basis of the RNN
    Args:
        rnn: RNN, RNN model
    Returns:
        U: torch.Tensor (N,tr), left singular vectors
        V: torch.Tensor (tr,N), (scaled) right singular vectors
        B: torch.Tensor (N,), biases

    """
    U = torch.clone(rnn.rnn.m.detach())
    N, tr = U.shape
    alpha = rnn.rnn.dt / rnn.rnn.tau
    V = torch.clone(rnn.rnn.n.detach() * alpha / N)
    W_or = U @ V
    U, s, V = torch.linalg.svd(W_or, full_matrices=False)
    U, s, V = U[:, :tr], s[:tr], V[:tr, :]
    V = (V.T * s).T
    if torch.linalg.norm(W_or - U @ V) > 1e-5:
        print("Warning: not orthogonal")
        print(torch.linalg.norm(W_or - U @ V))
    B = rnn.rnn.b_rec.detach()
    return U, V, B


def rotate_basis_vectors(vae, rotation):
    """
    Rotate the basis vectors of the vae.rnn
    Args:
        vae: VAE, VAE model
        rotation: torch.Tensor (dim_z,dim_z), transformation matrix
    Returns:
        vae: VAE, VAE model with rotated basis vectors
    """

    with torch.no_grad():
        m_or = vae.rnn.transition.m  # 20,2
        n_or = vae.rnn.transition.n
        m_new = m_or @ np.linalg.inv(rotation)
        n_new = rotation @ n_or
        if vae.rnn.params["scalar_noise_z"] == "Cov":
            proj_chol = vae.rnn.chol_cov_embed(vae.rnn.R_z)
        else:
            proj_chol = torch.diag(vae.rnn.std_embed_z(vae.rnn.R_z))
        proj_chol = rotation @ proj_chol
        vae.rnn.transition.m.copy_(m_new)
        vae.rnn.transition.n.copy_(n_new)
        vae.rnn.chol_cov_embed = lambda x: torch.tril(x)
        vae.rnn.R_z = torch.nn.Parameter(torch.linalg.cholesky(proj_chol @ proj_chol.T))
        vae.rnn.params["scalar_noise_z"] = "Cov"
    return vae


def get_loadings(vae):
    """
    Extract the loadings of the vae.rnn
    Args:
        vae: VAE, VAE model
    Returns:
        tau: np.array (dim_z,), time constants
        pV: np.array (dim_z,dim_x), right singular vectors
        pU: np.array (dim_x,dim_z), (scaled) left singular vectors
        pB: np.array (dim_x,), biases
        pI: np.array (dim_x,), input weights

    """
    prior = vae.rnn.transition
    tau = prior.cast_A(prior.AW).detach().numpy().squeeze()
    pV = (prior.n * prior.scaling).detach().numpy()
    pU = prior.m_transform(prior.m).detach().numpy()
    pB = prior.h.detach().numpy()
    pI = prior.Wu.detach().numpy()
    return tau, pV, pU, pB, pI


def get_orth_proj_latents(vae):
    """
    For projection latents on orthogonalised basis
    """
    with torch.no_grad():
        m_or = vae.rnn.transition.m
        n_or = vae.rnn.transition.n
        J = m_or @ n_or
        u, s, v = torch.linalg.svd(J)
        projection_matrix = u[:, : vae.dim_z].T @ m_or
    return projection_matrix


def convert_clipped(V, U, h):
    # We can map a clipped relu network to a regular clipped network
    # see  https://proceedings.mlr.press/v162/brenner22a/brenner22a.pdf 6.4.5. PROOF OF PROPOSITION 1
    """Returns equivalent model where
        max(x-h,0) is substituted for max(x+h,0)-max(x,0)
        
    Args:
        V: numpy array of shape (N,R)
        U: numpy array of shape (N,R)
        h: numpy array of shape (N,)

    Returns:
        V: numpy array of shape (Nx2,R)
        U: numpy array of shape (Nx2,R)
        h: numpy array of shape (Nx2,)
    """
    N = V.shape[0]
    h_new = np.zeros(N*2)
    h_new[:N] = -h
    V_new = np.concatenate([V,-V],axis=0)
    U_new = np.concatenate([U,U],axis=0)
    return V_new,U_new,h_new


def test_conversion(V, V_new, U, U_new, h, h_new, N, R, neuromodulation=None, s=None):
    x = np.random.randn(N)
    z = np.random.randn(R)
    x_n = np.concatenate([x,x])
    s = np.random.randn(N)
    relu = lambda x: np.maximum(x,0)
    if neuromodulation is not None: 
        W_new = np.diag(np.concatenate([s, s]))
        W = np.diag(s)

    if neuromodulation == 'postsynaptic':
        # test Z
        z_n = V_new.T@W_new@relu(U_new@z-h_new) 
        z_n2 = V@W@(relu(U@z+h)-relu(U@z))
        print(np.linalg.norm(z_n-z_n2))
        
        # test X
        x2 = U@V@(relu(x+h)-relu(x))
        J = U_new@V_new.T
        x3 = J@(relu(x_n-h_new))
        x3 = x3[:N]#+x3[N:]
        print(np.linalg.norm(x2-x3))
    
    elif neuromodulation == 'presynaptic':
        # test Z
        z_n = V_new.T@relu(W_new@U_new@z-h_new) 
        z_n2 = V@(relu(W@U@z+h)-relu(W@U@z))
        print(np.linalg.norm(z_n-z_n2))
        
        # test X
        x2 = U@V@(relu(W_new@x+h)-relu(W_new@x))
        J = U_new@V_new.T
        x3 = J@(relu(W@x_n-h_new))
        x3 = x3[:N]#+x3[N:]
        print(np.linalg.norm(x2-x3))


def relu_derivative(x):
    return np.array(x>0).astype('float')


def relu(x):
    return np.maximum(x, 0)


def F_z(z, U, V, hz, a, A, s):
    N = U.shape[0]
    I = np.eye(N)
    W = I + np.diag(A @ s)
    x = U @ z
    phi = relu(x + hz) - relu(x)
    return a * z + V.T @ W @ phi


def compute_jacobian(z, U, V, hz, a, A, s, neuromodulation=None):
    N,R = U.shape
    I = np.eye(N)

    if A is not None and neuromodulation == 'rank':
        W = np.eye(R) + np.diag(A @ s) 
    elif A is not None and neuromodulation != 'additive':
        W = I + np.diag(A @ s)
    elif A is not None: 
        W = A @ s
        
    if A is not None and neuromodulation == 'postsynaptic':
        x = U @ z
        phi_drv = np.diag(relu_derivative(x + hz)) - np.diag(relu_derivative(x))
        W = I + np.diag(A @ s)
        J = a * np.eye(R) + V @ W @ phi_drv @ U
        
    elif A is not None and neuromodulation == 'presynaptic':
        W = I + np.diag(A @ s) 
        x = W @ U @ z
        phi_drv = np.diag(relu_derivative(x + hz)) - np.diag(relu_derivative(x))
        J = a * np.eye(R) + V @ W @ phi_drv @ U

    elif A is not None and neuromodulation == 'rank': 
        W = np.eye(R) + np.diag(A @ s) 
        x = U @ z 
        phi_drv = np.diag(relu_derivative(x + hz)) - np.diag(relu_derivative(x))
        J = a * np.eye(R) + W @ V @ phi_drv @ U
    
    else:
        x = U @ z
        phi_drv = np.diag(relu_derivative(x + hz)) - np.diag(relu_derivative(x))
        J = a * np.eye(R) + V @ phi_drv @ U
        
    return J 


def fixed_point_sweep(V_new, U_new, hz, h_new, a, A, W_inp, d=2, neuromodulation='postsynaptic', start=0.05, end=1.0, num_points=200):
    s_vals = np.linspace(start, end, num_points)
    fixed_points = []
    inp = np.zeros((9,))
    rank = U_new.shape[1]
    for i, s_val in enumerate(s_vals):
        print(i, s_val)
        # find fixed point for given value of s
        a_arr = np.full((rank,), a)
        D_list,D_inds,z_list, n_singular = find_fixed_points_analytic(a_arr, V_new.T, U_new, hz, h_new, d=d, neuromodulation=neuromodulation, A=A, s=np.array([s_val]), Wu=W_inp, u=inp)
        fixed_points.append(z_list)
    return s_vals, fixed_points


def gmm_fit(neurons_fs, n_components, algo='bayes', n_init=50, random_state=None, mean_precision_prior=None,
            weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None):
    """
    fit a mixture of gaussians to a set of vectors
    :param neurons_fs: list of numpy arrays of shape n or numpy array of shape n x d
    :param n_components: int
    :param algo: 'em' or 'bayes'
    :param n_init: number of random seeds for the inference algorithm
    :param random_state: random seed for the rng to eliminate randomness
    :return: vector of population labels (of shape n), best fitted model
    """
    if isinstance(neurons_fs, list):
        X = np.vstack(neurons_fs).transpose()
    else:
        X = neurons_fs
    if algo == "em":
        model = GaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state)
    else:
        model = BayesianGaussianMixture(n_components=n_components, n_init=n_init, random_state=random_state,
                                        init_params='random', mean_precision_prior=mean_precision_prior,
                                        weight_concentration_prior_type=weight_concentration_prior_type,
                                        weight_concentration_prior=weight_concentration_prior)
    model.fit(X)
    z = model.predict(X)
    return z, model


def classify_fixed_points(U, V, h, a, A, s_vals, fixed_points, neuromodulation):
    stable = []
    unstable = []
    for i in range(len(s_vals)):
        for j in range(len(fixed_points[i])):
            # compute the stability of each fixed point 
            z_cur = fixed_points[i][j]
            J = compute_jacobian(z_cur, U, V, h, a, A, np.array([s_vals[i]]), neuromodulation)
            e, v = np.linalg.eig(J)
            if (e < 1.0).all():
                stable.append([z_cur[0], z_cur[1], s_vals[i]])
            else:
                unstable.append([z_cur[0], z_cur[1], s_vals[i]])
    return stable, unstable


def generate_trajectory(x_test, s_test, stim, vae, neuromod, noise_scale=1.0, sim_s=False):
    """
    Args:
        x_test (torch.tensor; neurons x time): Neural activity 
        s_test (torch.tensor; time); Neuromodulation signal
        stim (torch.tensor; number of stimuli x time): Stimuli traces
        vae (VAE): the model 
        neuromod: the neuromodulation type
        noise_scale: How much noise to add to the trajectories
    Returns:
        obs (torch.tensor; neurons x T): Generated trajectory
    """
        
    T = s_test.shape[0]
    s = s_test.view(1, 1, -1)
    if stim is not None:
        u = stim.unsqueeze(0) 
    
    # Retrieve initial latent state by projecting x onto z
    z_hat, _, _, _ = vae.encoder(x_test.unsqueeze(0))
    z0 = z_hat[:, :, 0].squeeze()
    if stim is not None:
        sim_v = True
    else:
        stim=None 
        sim_v = False

    if neuromod is None: 
        if stim is None:
            Z = vae.rnn.get_latent_time_series(time_steps=T, cut_off=0, z0=z0, u=stim, noise_scale=noise_scale, sim_v=sim_v, sim_s=False)
            obs = vae.rnn.get_observation(Z, noise_scale=0)[0, :, :, 0]
        else:   
            Z, V = vae.rnn.get_latent_time_series(time_steps=T, cut_off=0, z0=z0, u=u, noise_scale=noise_scale, sim_v=sim_v, sim_s=False)
            obs = vae.rnn.get_observation(Z, v=V, noise_scale=0)[0, :, :, 0]
    elif 'additive' in neuromod or sim_s==True:
        if stim is None:
            Z, S = vae.rnn.get_latent_time_series(time_steps=T, cut_off=0, z0=z0, s=s, noise_scale=noise_scale, sim_v=sim_v, sim_s=True)
            obs = vae.rnn.get_observation(Z, s=S, noise_scale=0)[0, :, :, 0]
        else:
            Z, V, S = vae.rnn.get_latent_time_series(time_steps=T, cut_off=0, z0=z0, u=u, s=s, noise_scale=noise_scale, sim_v=sim_v, sim_s=True)
            obs = vae.rnn.get_observation(Z, s=S, v=V, noise_scale=0)[0, :, :, 0]
    else: 
        if stim is None:
            Z = vae.rnn.get_latent_time_series(time_steps=T, cut_off=0, z0=z0, u=stim, s=s, noise_scale=noise_scale, sim_v=sim_v, sim_s = False)
            obs = vae.rnn.get_observation(Z, noise_scale=0)[0, :, :, 0]
        else:
            Z, V = vae.rnn.get_latent_time_series(time_steps=T, cut_off=0, z0=z0, u=stim if stim is None else u, s=s, noise_scale=noise_scale, sim_v=sim_v, sim_s = False)
            obs = vae.rnn.get_observation(Z, v=V, noise_scale=0)[0, :, :, 0]
    if sim_s: 
        return (z_hat.detach(), S.detach(), obs.detach())
    return (z_hat.detach(), obs.detach()) 