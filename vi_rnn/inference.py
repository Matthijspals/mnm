import torch
import numpy as np
from initialize_parameterize import chol_cov_embed


def filtering_posterior_optimal_proposal(
    vae, x, u, k=1, resample="systematic", t_forward=0, ed_ratio=0.0, encoder_padding=0
):
    """
    Forward pass of the VAE
    Note, here the approximate posterior is the optimal linear combination of the encoder and the RNN
    This can be calculated using the Kalman filter for linear observations and non-linear latents
    Args:
        x (torch.tensor; n_trials x dim_X x time_steps): input data
        u (torch.tensor; n_trials x dim_U x time_steps): input stim
        k (int): number of particles
        resample (str): resampling method
        t_forward (int): number of time steps to predict forward without using the data

    Returns:
        log_likelihood (torch.tensor; n_trials): log likelihood (with averaging over particles in the log)
        Qzs (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the approximate posterior
        alphas  (torch.tensor; n_trials x dim_z x dim_z x time_steps): interpolation coefficients
    """
    log_ws = []
    Qzs = []
    alphas = []

    ll_x_func = vae.rnn.get_observation_log_likelihood

    if resample == "multinomial":
        resample_f = resample_multinomial
    elif resample == "systematic":
        resample_f = resample_systematic
    elif resample == "none":
        resample_f = lambda x: x
    else:
        ValueError("resample does not exist, use one of: multinomial, systematic, none")

    batch_size, dim_x, time_steps = x.shape

    #print(ed_ratio)
    # Create encoder dropout mask
    if ed_ratio>0:
        ed_mask = torch.rand(batch_size, time_steps, device=x.device) > ed_ratio # True means use encoder
    else: 
        ed_mask = torch.ones(batch_size, time_steps, device=x.device, dtype=torch.bool)


    dim_z = vae.dim_z
    x = x.unsqueeze(-1)  # add particle dimension

    # project and clamp the variances
    eff_var_transition_chol = chol_cov_embed(vae.rnn.R_z)
    eff_var_transition_t0_chol = chol_cov_embed(vae.rnn.R_z_t0)

    eff_var_x = torch.clip(vae.rnn.var_embed_x(vae.rnn.R_x), vae.min_var)
    eff_std_x = torch.clip(vae.rnn.std_embed_x(vae.rnn.R_x), np.sqrt(vae.min_var))

    # set Kalman update function
    if dim_x < dim_z * 4:
        kalman_update = Kalman_update_highD
    else:
        kalman_update = Kalman_update_lowD

    # Get the initial transition mean
    if vae.rnn.simulate_input:
        v = torch.zeros(batch_size, vae.dim_u, 1, device=x.device)
    else:
        v = u[:, :, 0].unsqueeze(-1)  # add particle dimension



    transition_mean = (
        vae.rnn.get_initial_state(v[:, :, 0])
        .unsqueeze(2)
        .expand(batch_size, vae.dim_z, k)
    )

    # Get the observation weights and bias
    if vae.rnn.params["readout_from"] == "currents":
        m = vae.rnn.transition.m
        B = vae.rnn.observation.B.unsqueeze(-1) * m[:vae.dim_x] #if N>dim_x
        B = B.T
    elif vae.rnn.params["readout_from"] == "z_and_v":
        B = vae.rnn.observation.B[vae.dim_u:]
        Bu = vae.rnn.observation.B[:vae.dim_u]
        
    else:
        B = vae.rnn.observation.B
    Obs_bias = vae.rnn.observation.Bias.view(1, -1, 1)

    # Get Kalman gain
    alpha, one_min_alpha, Kalman_gain, var_Q_cholesky = kalman_update(
        eff_var_transition_t0_chol, B, eff_var_x
    )
    

    # Posterior Mean
    mean_Q = torch.einsum("zs,BsK->BzK", one_min_alpha, transition_mean) + torch.einsum(
        "zx,BxK->BzK", Kalman_gain, x[:, :, 0] - Obs_bias
    )

    # Sample from posterior and calculate likelihood
    Q_dist = torch.distributions.MultivariateNormal(
        loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky
    )
    Qz = Q_dist.rsample()
    ll_qz = Q_dist.log_prob(Qz)

    # Calculate likelihood under the transition
    pz_dist = torch.distributions.MultivariateNormal(
        loc=transition_mean.permute(0, 2, 1), scale_tril=eff_var_transition_t0_chol
    )
    ll_pz = pz_dist.log_prob(Qz)

    # Get observation mean and calculate likelihood of the data
    Qz = Qz.permute(0, 2, 1)

    mean_x = torch.einsum("zx, bzk -> bxk", B, Qz) + Obs_bias
    ll_x = ll_x_func(x[:, :, 0], mean_x)

    # Calculate the log weights
    log_w = ll_x + ll_pz - ll_qz
    Qz, indices = resample_f(Qz, log_w)

    # Store some quantities
    log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
    Qzs.append(Qz)

    time_steps = x.shape[2] - t_forward
    u = u.unsqueeze(-1)  # account for k

    # Precalculate Kalman Gain / Interpolation
    alpha, one_min_alpha, Kalman_gain, var_Q_cholesky = kalman_update(
        eff_var_transition_chol, B, eff_var_x
    )
    # Start the loop through the time steps
    time_steps -= encoder_padding

    for t in range(1, time_steps):
        # Get the transition mean
        transition_mean = vae.rnn.transition(Qz, v=v)
        # progress input dynamics
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t - 1])
        else:
            v = u[:, :, t]

        if vae.rnn.params["readout_from"] == "currents":
            v_to_X = torch.einsum("xv, bvk -> bxk", vae.rnn.transition.Wu, v)[:,:vae.dim_x]
        elif vae.rnn.params["readout_from"] == "z_and_v":
            v_to_X = torch.einsum("vx, bvk -> bxk", Bu, v)
        else:
            v_to_X = 0

        x_t = x[:, :, t] - Obs_bias - v_to_X
        mean_Q = torch.einsum(
            "zs,BsK->BzK", one_min_alpha, transition_mean
        ) + torch.einsum("zx,BxK->BzK", Kalman_gain, x_t)

        # add mask
        mask = ed_mask[:,t].view(-1,1,1)
        mean_Q = mask * mean_Q + (~mask) * transition_mean
        #print( mask * mean_Q)
        #print((~mask) * transition_mean)
        #eff_var_transition_chol_expanded = eff_var_transition_chol.unsqueeze(0).expand(batch_size, -1, -1)
        #print(var_Q_cholesky.shape, eff_var_transition_chol.shape)
        var_Q_cholesky_masked = mask * var_Q_cholesky + (~mask) * eff_var_transition_chol#_expanded
        """
        print(var_Q_cholesky_masked.shape)
        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.MultivariateNormal(
            loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky_masked
        )
        Qz = Q_dist.rsample()
        ll_qz = Q_dist.log_prob(Qz)
        """
        mean_Q_flat = mean_Q.permute(0, 2, 1).reshape(batch_size * k, dim_z)          # (B*K, z)
        var_Q_cholesky_flat = var_Q_cholesky_masked.repeat_interleave(k, dim=0) # (B*K, z, z)

        # ---- Define Gaussian and sample ----
        Q_dist = torch.distributions.MultivariateNormal(
            loc=mean_Q_flat,
            scale_tril=var_Q_cholesky_flat
        )

        Qz = Q_dist.rsample()            # (B*K, z)
        ll_qz = Q_dist.log_prob(Qz)      # (B*K,)

        # ---- Reshape back to (B, z, K) and (B, K) ----
        Qz = Qz.view(batch_size, k, dim_z)#.permute(0, 2, 1)  # (B, z, K)
        ll_qz = ll_qz.view(batch_size, k)




        # Calculate likelihood under the transition
        pz_dist = torch.distributions.MultivariateNormal(
            loc=transition_mean.permute(0, 2, 1), scale_tril=eff_var_transition_chol
        )
        ll_pz = pz_dist.log_prob(Qz)

        # Get observation mean and calculate likelihood of the data
        Qz = Qz.permute(0, 2, 1)
        mean_x = torch.einsum("zx, bzk -> bxk", B, Qz) + Obs_bias + v_to_X

        ll_x = ll_x_func(x[:, :, t], mean_x)

        # Calculate the log weights
        log_w = ll_x + ll_pz - ll_qz
        #print(torch.mean(ll_x),torch.mean(ll_pz),torch.mean(ll_qz))
        Qz, indices = resample_f(Qz, log_w)

        # Note: weights also have analytic expression:
        # https://www.ecmwf.int/sites/default/files/elibrary/2012/76468-particle-filters-optimal-proposal-and-high-dimensional-systems_0.pdf
        # w_mean = torch.einsum("zx, bzk -> bxk", B, transition_mean) + Obs_bias + v_to_X
        # w_upd =  torch.einsum("zx, zs, sy -> xy", B, eff_var_transition, B)+torch.diag(eff_var_x)
        # w_chol = torch.linalg.cholesky(w_upd)
        # w_dist = torch.distributions.MultivariateNormal(loc=w_mean.permute(0, 2, 1), scale_tril=w_chol#)
        # ll_w = w_dist.log_prob(x[:, :, t].permute(0, 2, 1))

        # Store some quantities
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)
        alphas.append(alpha)

    # Use transition samples for the last t_forward steps
    for t in range(time_steps, time_steps + t_forward):

        # Here transition and posterior are the same and we just need the likelihood of the data
        transition_mean = vae.rnn.transition(Qz, v=v).squeeze(2)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t - 1])
        else:
            v = u[:, :, t]

        if vae.rnn.params["readout_from"] == "currents":
            v_to_X = torch.einsum("xv, bvk -> bxk", vae.rnn.transition.Wu, v)
        elif vae.rnn.params["readout_from"] == "z_and_v":
            v_to_X = torch.einsum("vx, bvk -> bxk", Bu, v)
        else:
            v_to_X = 0

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.MultivariateNormal(
            loc=transition_mean.permute(0, 2, 1), scale_tril=eff_var_transition_chol
        )
        Qz = Q_dist.rsample().permute(0, 2, 1)
        mean_x = torch.einsum("zx, bzk -> bxk", B, Qz) + Obs_bias + v_to_X
        ll_x = ll_x_func(x[:, :, t], mean_x)
        log_w = ll_x

        # Store some quantities
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)
    # Make tensors from lists
    log_ws = torch.stack(log_ws)
    alphas = torch.stack(alphas)

    # Average over time steps
    log_likelihood = log_ws
    Qzs = torch.stack(Qzs)
    Qzs = Qzs.permute(1, 2, 0, 3)

    return (
        log_likelihood,
        Qzs,
        alphas,
    )


def filtering_posterior(
    vae,
    x,
    u=None,
    k=1,
    resample="systematic",
    t_forward=0,
    ed_ratio=0.0,
    encoder_padding=0,
):
    """
    Forward pass of the VAE
    Note, here the approximate posterior is a linear combination of the encoder and the RNN
    Args:
        x (torch.tensor; n_trials x dim_X x time_steps): input data
        u (torch.tensor; n_trials x dim_U x time_steps): input stim
        k (int): number of particles
        resample (str): resampling method
        t_forward (int): number of time steps to predict forward without using the data
    Returns:
        log_likelihood (torch.tensor; n_trials): log likelihood (with averaging over particles in the log)
        Qzs (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the approximate posterior
        alphas  (torch.tensor; n_trials x dim_z x dim_z x time_steps): interpolation coefficients
    """
    batch_size = x.shape[0]
    ll_x_func = vae.rnn.get_observation_log_likelihood

    if resample == "multinomial":
        resample_f = resample_multinomial
    elif resample == "systematic":
        resample_f = resample_systematic
    elif resample == "none":
        resample_f = lambda x: x
    else:
        ValueError("resample does not exist, use one of: multinomial, systematic, none")


    # Run the encoder
    mean_enc, log_var_enc = vae.encoder(
        x[:, : vae.dim_x, : x.shape[2] - t_forward], k=k
    )  # Bs,Dx,T,K
    bs, dim_z, time_steps, _ = mean_enc.shape
    # Create encoder dropout mask
    if ed_ratio>0:
        ed_mask = torch.rand(batch_size, time_steps, device=x.device) > ed_ratio
    else: 
        ed_mask = torch.ones(1, time_steps, device=x.device)

    # Project and clamp the variances
    eff_var_enc = torch.clamp(torch.exp(log_var_enc), min=vae.min_var, max=vae.max_var)

    eff_var_transition = torch.clamp(
        vae.rnn.var_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=vae.min_var,
        max=vae.max_var,
    )  # 1,Dz,1
    eff_std_transition = torch.clamp(
        vae.rnn.std_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1
    eff_var_transition_t0 = torch.clamp(
        vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=vae.min_var,
        max=vae.max_var,
    )  # 1,Dz,1
    eff_std_transition_t0 = torch.clamp(
        vae.rnn.std_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1
    #print(log_var_enc.mean(), "{:.5f}".format(torch.exp(log_var_enc).mean()), "{:.5f}".format(eff_var_transition.mean()), "{:.5f}".format(eff_var_transition_t0.mean()))

    # Cut some of the data if a CNN was used without padding

    x_hat = x.unsqueeze(-1)

    if vae.rnn.observation.has_state:
        vae.rnn.observation.set_initial_state(batch_size, k)

    # Initialise some lists
    log_ws = []
    Qzs = []
    alphas = []

    # Get the initial transition mean
    if vae.rnn.simulate_input:
        v = torch.zeros(batch_size, vae.dim_u, 1, device=x.device)
    else:
        v = u[:, :, 0].unsqueeze(-1)  # add particle dimension

    transition_mean = (
        vae.rnn.get_initial_state(v[:, :, 0])
        .unsqueeze(2)
        .expand(batch_size, vae.dim_z, k)
    )

    # Calculate the initial posterior mean and covariance
    Qz, ll_qz, alpha = diagonal_proposal(
        eff_var_transition_t0, eff_var_enc[:, :, 0], transition_mean, mean_enc[:, :, 0], mask=torch.ones_like(ed_mask[:,0])
    )

    # Calculate the log likelihood under the transition
    ll_pz = (
        torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition_t0)
        .log_prob(Qz)
        .sum(axis=1)
    )

    # Get the observation mean and calculate likelihood of the data
    mean_x = vae.rnn.observation(Qz, v=v)
    ll_x = ll_x_func(x_hat[:, :, 0], mean_x)

    # Calculate the log weights
    log_w = ll_x + ll_pz - ll_qz
    Qz, indices = resample_f(Qz, log_w)
    if vae.rnn.observation.has_state:
        vae.rnn.observation.resample_state(indices)
    # Store some quantities
    alphas.append(alpha)
    log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
    Qzs.append(Qz)

    u = u.unsqueeze(-1)  # add particle dimension

    time_steps -= encoder_padding
    # Loop through the time steps
    for t in range(1, time_steps):

        # Get the transition mean
        transition_mean = vae.rnn.transition(Qz, v=v)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t - 1])
        else:
            v = u[:, :, t]

        # Calculate the posterior mean and covariance
        Qz, ll_qz, alpha = diagonal_proposal(
            eff_var_transition, eff_var_enc[:, :, t], transition_mean, mean_enc[:, :, t], mask=ed_mask[:,t]
        )

        # Calculate the log likelihood under the transition
        ll_pz = (
            torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition)
            .log_prob(Qz)
            .sum(axis=1)
        )

        # Get the observation mean and calculate likelihood of the data
        mean_x = vae.rnn.observation(Qz, v=v)
        ll_x = ll_x_func(x_hat[:, :, t], mean_x)

        # Calculate the log weights
        log_w = ll_x + ll_pz - ll_qz

        # Store some quantities
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qz, indices = resample_f(Qz, log_w)
        if vae.rnn.observation.has_state:
            vae.rnn.observation.resample_state(indices)

        Qzs.append(Qz)
        alphas.append(alpha)

    # Use transition model for the last t_forward steps
    for t in range(time_steps, time_steps + t_forward):

        # Here transition and posterior are the same and we just need the likelihood of the data
        transition_mean = vae.rnn.transition(Qz, v=v).squeeze(2)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t - 1])
        else:
            v = u[:, :, t]

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=torch.sqrt(eff_var_transition)
        )
        Qz = Q_dist.rsample()
        mean_x = vae.rnn.observation(Qz, v=v)
        ll_x = ll_x_func(x_hat[:, :, t], mean_x)
        log_w = ll_x

        # Store some quantities
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)

    # Make tensors from lists
    log_ws = torch.stack(log_ws)
    alphas = torch.stack(alphas)

    # Average over time steps
    #log_likelihood = torch.mean(log_ws, axis=0)
    log_likelihood = log_ws
    Qzs = torch.stack(Qzs)
    Qzs = Qzs.permute(1, 2, 0, 3)

    return log_likelihood, Qzs, alphas


def Kalman_update_lowD(eff_var_transition_chol, B, eff_var_x):
    """perform Kalman update step, efficient when dim_z << dim_x
    
    Args:
        eff_var_transition_chol: (z, z) Cholesky of transition covariance
        B: (z, x) observation mapping
        eff_var_x: (x,) effective observation variance (diag)
        mask: optional (B,) boolean tensor indicating whether to apply update
    """
    dim_z = eff_var_transition_chol.shape[0]
    eff_var_x_inv = 1.0 / eff_var_x

    var_Q = torch.linalg.inv(
        torch.cholesky_inverse(eff_var_transition_chol)
        + (B * torch.unsqueeze(eff_var_x_inv, 0)) @ B.T
    )
    Kalman_gain = var_Q @ (B * torch.unsqueeze(eff_var_x_inv, 0))
    alpha = Kalman_gain @ B.T
    one_min_alpha = torch.eye(dim_z, device=alpha.device) - alpha

    var_Q = torch.eye(dim_z, device=alpha.device) * 1e-8 + (var_Q + var_Q.T) / 2
    var_Q_cholesky = torch.linalg.cholesky(var_Q)

    return alpha, one_min_alpha, Kalman_gain, var_Q_cholesky


def Kalman_update_highD(eff_var_transition_chol, B, eff_var_x, mask=None):
    """Perform Kalman update step; optionally mask updates (skip when mask=False).
    
    Args:
        eff_var_transition_chol: (z, z) Cholesky of transition covariance
        B: (z, x) observation mapping
        eff_var_x: (x,) effective observation variance (diag)
        mask: optional (B,) boolean tensor indicating whether to apply update
    """
    #print(eff_var_transition_chol.shape, B.shape, eff_var_x.shape)
    dim_z = eff_var_transition_chol.shape[0]

    eff_var_transition = eff_var_transition_chol @ eff_var_transition_chol.T
    Kalman_gain = (
        eff_var_transition
        @ B
        @ torch.linalg.inv(torch.diag(eff_var_x) + B.T @ eff_var_transition @ B)
    )
    alpha = Kalman_gain @ B.T
    one_min_alpha = torch.eye(dim_z, device=alpha.device) - alpha

    # Posterior Joseph stabilized Covariance
    var_Q = (
        one_min_alpha @ eff_var_transition @ one_min_alpha.T
        + (Kalman_gain * torch.unsqueeze(eff_var_x, 0)) @ Kalman_gain.T
    )
    var_Q = torch.eye(dim_z, device=alpha.device) * 1e-8 + (var_Q + var_Q.T) / 2
    var_Q_cholesky = torch.linalg.cholesky(var_Q)

    return alpha, one_min_alpha, Kalman_gain, var_Q_cholesky


def diagonal_proposal(eff_var_transition, E_var, transition_mean, E_mean, mask=1.0):
    """
    Computes diagonal covariance of the proposal
    Args:
        eff_var_transition (torch.tensor; 1 x dim_z x 1): transition variance
        E_var (torch.tensor; n_trials x dim_z x 1): encoder variance
        transition_mean (torch.tensor; n_trials x dim_z x k): transition mean
        E_mean (torch.tensor; n_trials x dim_z x k): encoder mean
    Returns:
        Qz (torch.tensor; n_trials x dim_z x k): posterior samples
        ll_qz (torch.tensor; n_trials x k): log likelihood of the posterior
        alpha (torch.tensor; n_trials x dim_z x 1): interpolation coefficients

    """
    precZ = 1 / eff_var_transition
    precE = 1 / E_var
    #print("Evar transition")
    #print(E_var.mean())
    #print(eff_var_transition.mean())
    precQ = precZ + precE * mask.view(-1,1,1)
    alpha = 1 - precZ / precQ
    eff_var_Q = 1 / precQ
    mean_Q = (precZ * transition_mean + precE * E_mean) * eff_var_Q
    Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
    Qz = Q_dist.rsample()
    ll_qz = Q_dist.log_prob(Qz).sum(axis=1)
    #print(alpha.mean())
    return Qz, ll_qz, alpha


def filtering_posterior_bootstrap(
    vae,
    x,
    u=None,
    k=1,
    resample="systematic",
    t_forward=0,
    ed_ratio=0.0,
):
    """
    Forward pass of the VAE
    Note, here the approximate posterior is just the RNN
    Args:
        x (torch.tensor; n_trials x dim_X x time_steps): input data
        u (torch.tensor; n_trials x dim_U x time_steps): input stim
        k (int): number of particles
        resample (str): resampling method
        t_forward (int): number of time steps to predict forward without using the data
    Returns:
        log_likelihood (torch.tensor; n_trials): log likelihood (with averaging over particles in the log)
        Qzs (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the approximate posterior
        alphas  (torch.tensor; n_trials x dim_z x dim_z x time_steps): interpolation coefficients

    """
    if resample == "multinomial":
        resample_f = resample_multinomial
    elif resample == "systematic":
        resample_f = resample_systematic
    elif resample == "none":
        resample_f = lambda x: x
    else:
        ValueError("resample does not exist, use one of: multinomial, systematic, none")

    # Define the data likelihood function
    ll_x_func = vae.rnn.get_observation_log_likelihood

    # Project and clamp the variances
    eff_std_transition = torch.clamp(
        vae.rnn.std_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1
    eff_std_transition_t0 = torch.clamp(
        vae.rnn.std_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1

    x_hat = x.unsqueeze(-1)

    # Initialise some lists
    log_ws = []
    log_ll = []
    Qzs = []

    # Get the initial transition mean
    batch_size, dim_x, time_steps = x.shape

    # Get the initial transition mean
    if vae.rnn.simulate_input:
        v = torch.zeros(batch_size, vae.dim_u, 1, device=x.device)
    else:
        v = u[:, :, 0].unsqueeze(-1)  # add particle dimension

    transition_mean = (
        vae.rnn.get_initial_state(v[:, :, 0])
        .unsqueeze(2)
        .expand(batch_size, vae.dim_z, k)
    )

    # Calculate the initial posterior mean and covariance

    # Sample from the posterior and calculate likelihood
    Q_dist = torch.distributions.Normal(
        loc=transition_mean, scale=eff_std_transition_t0
    )
    Qz = Q_dist.rsample()

    # Get the observation mean and calculate likelihood of the data
    mean_x = vae.rnn.observation(Qz, v=v)
    ll_x = ll_x_func(x_hat[:, :, 0], mean_x)

    # Calculate the log weights and resample
    log_w = ll_x
    Qz, indices = resample_f(Qz, log_w)

    # Store some quantities
    log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
    log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
    Qzs.append(Qz)

    u = u.unsqueeze(-1)  # add particle dimension

    # Loop through the time steps
    for t in range(1, time_steps + t_forward):

        # Get the transition mean
        transition_mean = vae.rnn.transition(Qz, v=v)
   

        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t - 1])
        else:
            v = u[:, :, t]

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=eff_std_transition
        )
        Qz = Q_dist.rsample()

        # Get the observation mean and calculate likelihood of the data
        mean_x = vae.rnn.observation(Qz, v=v)
        ll_x = ll_x_func(x_hat[:, :, t], mean_x)

        # Calculate the log weights
        log_w = ll_x

        # resample as needed
        if t < time_steps:
            Qz, indices = resample_f(Qz, log_w)

        # Store some quantities
        log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)

    # Make tensors from lists
    log_ws = torch.stack(log_ws)
    log_ll = torch.stack(log_ll)

    # Average over time steps
    log_likelihood = torch.mean(log_ws, axis=0)

    Qzs = torch.stack(Qzs)
    Qzs = Qzs.permute(1, 2, 0, 3)
    empty = torch.ones(0, device=Qzs.device)
    return log_likelihood, Qzs, empty




def predict_NLB(
    vae,
    x,
    u=None,
    k=1,
    t_held_in=None,
    t_forward=0,
    t_bs=0,
    resample="systematic",
    marginal_smoothing=False,
):
    """
    Obtain filtering and smoothing posteriors given data and optionally input

    Args:
        x (torch.tensor; n_trials x dim_X x time_steps): input data
        u (torch.tensor; n_trials x dim_U x time_steps): input stim
        k (int): number of particles
        t_held_in (int): number of time steps where the data / encoder is used
        t_forward (int): number of time steps to predict forward without using the data
        t_bs (int): number of time steps to predict forward with the bootstrap proposal (no encoder)
        resample (str): resampling method
        marginal_smoothing (bool): whether or not to obtain the marginal latents

    Returns:
        Qzs_filt (torch.tensor; n_trials x dim_z x time_steps x k): filtered latent time series
        Qzs_smooth (torch.tensor; n_trials x dim_z x time_steps x k): smoothed latent time series
        Xs_filt (torch.tensor; n_trials x dim_x x time_steps x k): filtered observation time series
        Xs_smooth (torch.tensor; n_trials x dim_x x time_steps x k): smoothed observation time series
    """
    if resample == "multinomial":
        resample_f = resample_multinomial
    elif resample == "systematic":
        resample_f = resample_systematic
    elif resample == "none":
        resample_f = lambda x: x
    else:
        ValueError("resample does not exist, use one of: multinomial, systematic, none")

    if t_held_in is None:
        t_held_in = x.shape[2]
    # no need for gradients here
    vae.eval()

    # define the data likelihood function
    ll_x_func = vae.rnn.get_observation_log_likelihood

    # Run the encoder
    Emean, log_Evar = vae.encoder(x[:, : vae.dim_x, :t_held_in], k=k)  # Bs,Dx,T,K
    x_hat = x.unsqueeze(-1)

    # Project and clamp the variances
    Evar = torch.clamp(torch.exp(log_Evar), min=vae.min_var, max=vae.max_var)

    eff_var_transition = torch.clamp(
        vae.rnn.var_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=vae.min_var,
        max=vae.max_var,
    )  # 1,Dz,1
    eff_std_transition = torch.clamp(
        vae.rnn.std_embed_z(vae.rnn.R_z).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1
    eff_var_transition_t0 = torch.clamp(
        vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=vae.min_var,
        max=vae.max_var,
    )  # 1,Dz,1
    eff_std_transition_t0 = torch.clamp(
        vae.rnn.std_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
        min=np.sqrt(vae.min_var),
        max=np.sqrt(vae.max_var),
    )  # 1,Dz,1
    bs, dim_z, time_steps, _ = Emean.shape

    # Initialise some lists
    log_ws = []
    Qzs = []
    Qzs_filt = []

    # Get the transition mean and observation mean
    if u is None:
        u = torch.zeros(bs, vae.dim_u, time_steps, 1).to(x.device)
    else:
        u = u.unsqueeze(-1)  # add particle dimension

    # Get the initial transition mean
    if vae.rnn.simulate_input:
        v = torch.zeros(bs, vae.dim_u, 1, device=x.device)
    else:
        v = u[:, :, 0]
    transition_mean = (
        vae.rnn.get_initial_state(v[:, :, 0]).unsqueeze(2).expand(bs, vae.dim_z, k)
    )

    # get the posterior mean and covariance
    precZ = 1 / eff_var_transition_t0
    precE = 1 / Evar[:, :, 0]
    precQ = precZ + precE
    alpha = precE / precQ
    mean_Q = (1 - alpha) * transition_mean + alpha * Emean[:, :, 0]
    eff_var_Q = 1 / precQ

    # Sample from the posterior and calculate likelihood
    Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
    Qz = Q_dist.rsample()
    ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

    # Calculate the log likelihood under the transition
    ll_pz = (
        torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition_t0)
        .log_prob(Qz)
        .sum(axis=1)
    )

    # Get the observation mean and calculate likelihood of the data
    mean_x = vae.rnn.observation(Qz, v=v)
    ll_x = ll_x_func(x_hat[:, :, 0], mean_x[:, : x_hat.shape[1]])
    # Calculate the log weights
    log_w = ll_x + ll_pz - ll_qz
    vs = [v]

    # Append the log weights and log likelihoods
    log_ws.append(log_w)
    Qzs.append(Qz)

    # Loop through the time steps
    for t in range(1, t_held_in):
        Qz, indices = resample_f(Qz, log_w)
        Qzs_filt.append(Qz)

        # Get the transition mean
        transition_mean = vae.rnn.transition(Qz, v=v)
        transition_mean = transition_mean
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, u[:, :, t - 1])
        else:
            v = u[:, :, t]
        vs.append(v)

        # Calculate the posterior mean and covariance
        precZ = 1 / eff_var_transition
        precE = 1 / Evar[:, :, t]
        precQ = precZ + precE
        alpha = precE / precQ
        mean_Q = (1 - alpha) * transition_mean + alpha * Emean[:, :, t]
        eff_var_Q = 1 / precQ

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
        Qz = Q_dist.rsample()
        ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

        # Calculate the log likelihood under the transition
        ll_pz = (
            torch.distributions.Normal(loc=transition_mean, scale=eff_std_transition)
            .log_prob(Qz)
            .sum(axis=1)
        )

        # Get the observation mean and calculate likelihood of the data
        mean_x = vae.rnn.observation(Qz, v=v)
        ll_x = ll_x_func(x_hat[:, :, t], mean_x[:, : x_hat.shape[1]])

        # Calculate the log weights
        log_w = ll_x + ll_pz - ll_qz
        log_ws.append(log_w)
        Qzs.append(Qz)

    for t in range(t_held_in, t_held_in + t_bs):
        Qz, indices = resample_f(Qz, log_w)
        Qzs_filt.append(Qz)

        # Here transition and posterior are the same and we just need the likelihood of the data
        transition_mean = vae.rnn.transition(Qz, v=v)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, torch.zeros_like(v))
        vs.append(v)
        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=torch.sqrt(eff_var_transition)
        )
        Qz = Q_dist.rsample()

        mean_x = vae.rnn.observation(Qz, v=v)
        ll_x = ll_x_func(x_hat[:, :, 0], mean_x[:, : x_hat.shape[1]])
        log_w = ll_x
        ll_qz = ll_qz

        # Store some quantities
        Qzs.append(Qz)

    # resample last time steps
    Qz, indices = resample_f(Qz, log_w)
    Qzs_filt.append(Qz)

    # Backward Smoothing
    Qzs_sm = torch.zeros(
        t_held_in + t_forward,
        *torch.stack(Qzs).shape[1:],
        device=x.device,
        dtype=x.dtype
    )
    Qzs_sm[t_held_in - 1] = Qzs_filt[-1]

    # Start from the end and move backwards
    log_weights_backward = log_ws[-1]
    for t in range(t_held_in - 2, -1, -1):

        if marginal_smoothing:
            # Marginal smoothing, note this jas K^2 cost!

            transition_mean = vae.rnn(Qzs[t], noise_scale=0, u=vs[t - 1])
            probs_ij = (
                torch.distributions.Normal(
                    loc=transition_mean.unsqueeze(3),
                    scale=eff_std_transition.unsqueeze(3),
                )
                .log_prob(Qzs_sm[t + 1])
                .sum(axis=1)
            )
            log_denom = torch.logsumexp(
                log_ws[t].unsqueeze(-1) * probs_ij[:, :], axis=1
            )

            log_weight = log_ws[t]
            for i in range(k):
                log_nom_i = probs_ij[:, i]
                reweight_i = torch.logsumexp(
                    log_weights_backward + log_nom_i - log_denom, axis=1
                )
                log_weight[:, i] += reweight_i
            Qzs_sm[t], indices = resample_f(Qzs[t], log_weight)

        else:
            # Conditional smoothing
            transition_mean = vae.rnn.transition(Qzs[t], v=vs[t - 1])
            ll_pz = (
                torch.distributions.Normal(
                    loc=transition_mean, scale=eff_std_transition
                )
                .log_prob(Qzs_sm[t + 1])
                .sum(axis=1)
            )
            log_weights_reweighted = log_ws[t] + ll_pz

            # Resample based on the backward weights
            Qzs_sm[t], indices = resample_f(Qzs[t], log_weights_reweighted)

    # Use forward samples for the last n_forward steps

    for t in range(t_held_in, t_held_in + t_bs + t_forward):
        transition_mean = vae.rnn.transition(Qz, v=v)
        if vae.rnn.simulate_input:
            v = vae.rnn.transition.step_input(v, torch.zeros_like(v))
        vs.append(v)

        Q_dist = torch.distributions.Normal(
            loc=transition_mean, scale=torch.sqrt(eff_var_transition)
        )
        Qz = Q_dist.rsample()
        Qzs_filt.append(Qz)
        Qzs_sm[t] = Qz

    Qzs_filt = torch.stack(Qzs_filt).permute(1, 2, 0, 3)
    Qzs_sm = Qzs_sm.permute(1, 2, 0, 3)
    vs = torch.stack(vs).permute(1, 2, 0, 3)
    Xs_filt = vae.rnn.observation(Qzs_filt, v=vs)
    Xs_sm = vae.rnn.observation(Qzs_sm, v=vs)

    return Qzs_filt, Qzs_sm, Xs_filt, Xs_sm


def resample_Q(Qz, indices):
    """
    Batch resample
    Args:
        Qz (torch.tensor; BS, dim_z, K): input data
        indices (torch.tensor; BS, K): indices to resample
    Returns:
        Qz_resampled (torch.tensor; BS, dim_z, K): resampled data
    """
    return torch.gather(Qz, 2, indices.unsqueeze(1).expand(Qz.shape))


def sample_indices_systematic(log_weight):
    """Sample ancestral index using systematic resampling.
    From: https://github.com/tuananhle7/aesmc

    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        indices (torch.tensor; BS, K): sampled indices
    """

    if torch.sum(log_weight != log_weight).item() != 0:
        raise FloatingPointError("log_weight contains nan element(s)")

    batch_size, num_particles = log_weight.size()
    indices = torch.zeros(
        batch_size, num_particles, device=log_weight.device, dtype=torch.long
    )
    log_weight = log_weight.to(dtype=torch.double).detach()
    uniforms = torch.rand(
        size=[batch_size, 1], device=log_weight.device, dtype=log_weight.dtype
    )
    pos = (
        uniforms + torch.arange(0, num_particles, device=log_weight.device)
    ) / num_particles

    normalized_weights = torch.exp(
        log_weight - torch.logsumexp(log_weight, axis=1, keepdims=True)
    )

    cumulative_weights = torch.cumsum(normalized_weights, axis=1)
    # hack to prevent numerical issues
    max = torch.max(cumulative_weights, axis=1, keepdims=True).values
    cumulative_weights = cumulative_weights / max

    for batch in range(batch_size):
        indices[batch] = torch.bucketize(pos[batch], cumulative_weights[batch])

    return indices


def resample_multinomial(Qz, log_w):
    indices = sample_indices_multinomial(log_w)
    Qz = resample_Q(Qz, indices)
    return Qz, indices


def resample_systematic(Qz, log_w):
    indices = sample_indices_systematic(log_w)
    Qz = resample_Q(Qz, indices)
    return Qz, indices


def sample_indices_multinomial(log_w):
    """Sample ancestral index using multinomial resampling.
    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        indices (torch.tensor; BS, K): sampled indices
    """
    k = log_w.shape[1]
    log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
    w_tilde = log_w_tilde.exp().detach()  # +1e-5
    w_tilde = w_tilde / w_tilde.sum(1, keepdim=True)
    return torch.multinomial(w_tilde, k, replacement=True)  # m* numsamples


def norm_and_detach_weights(log_w):
    """
    Normalize and detach weights
    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        reweight (torch.tensor; BS, K): normalized weights
    """
    log_weight = log_w.detach()
    log_weight = log_weight - torch.max(log_weight, -1, keepdim=True)[0]  #
    reweight = torch.exp(log_weight)
    reweight = reweight / torch.sum(reweight, -1, keepdim=True)
    return reweight
