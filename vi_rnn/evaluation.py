import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from evaluation.kl_Gauss import calc_kl_from_data
from evaluation.pse import power_spectrum_helling
import torch
import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
from scipy.stats import zscore
from scipy.signal import convolve 

def predict_X(
        vae, 
        trial_dur,
        eval_data,
        task_input, 
        s,
        smoothing=20,
        cut_off=0,
        freq_cut_off=10000,
        sim_obs_noise=1,
        sim_latent_noise=1,
        smooth=False, 
        neuromodulation=False,
        sim_v=True,
        sim_s=False
):
    """
    Get predicted trajectories 

    Args: 
        vae (nn.Module): VAE model 
        trial_dur (int): Duration of a single trial 
        eval_data (torch.Tensor): population activity on test data 
        task_input (torch.Tensor): task input tensor 
        s (torch.tensor): neuromodulator signals  
        smoothing (int): smoothing param for the power spectrum 
        cut_off (int): cut off for the latent time series 
        freq_cut_off (int): cut off for the power spectrum 
        sim_obs_noise (float): latent noise scale 
        smooth (bool): whether to smooth the generated trajectory 

    Returns:
        trajectories (torch.tensor; n_trials, dim_x, trial_dur)
    """
    n_trials = eval_data.shape[0] if len(eval_data.shape) == 3 else 1
    with torch.no_grad(): 
        # Evaluating on one long trajectory? 
        if len(eval_data.shape) == 2: 
            # obtain an initial latent state 
            if sim_latent_noise > 1e-8: # take sample of encoder 
                z_hat, _, _, _ = vae.encoder(eval_data[:trial_dur]) 
            else: # take mean prediction of encoder 
                _, z_hat, _, _ = vae.encoder(eval_data[:trial_dur])
            z0 = z_hat[:, :, 1].squeeze() 
            # predict latent time series now that we have initial latent state 
            Z, v, S = vae.rnn.get_latent_time_series(time_steps=trial_dur, 
                                               cut_off=cut_off,
                                               z0=z0,
                                               u=task_input,
                                               s=s,
                                               noise_scale=sim_latent_noise,
                                               sim_v=sim_v,
                                               sim_s=sim_s)
        else:
            # Evaluate on multiple short trajectories (trials) 
            if sim_latent_noise > 1e-8: 
                z_hat, _, _, _ = vae.encoder(eval_data.permute(0, 2, 1))
            else: 
                _, z_hat, _, _ = vae.encoder(eval_data.permute(0, 2, 1))
            z0 = z_hat[:, :, :1]
            Z, v, S = vae.rnn.get_latent_time_series(
                time_steps=trial_dur,
                cut_off=cut_off,
                z0=z0, 
                u=task_input,
                s=s,
                noise_scale=sim_latent_noise,
                sim_v=sim_v,
                sim_s=sim_s
            )
        # transform latent time series into observations 
        trajectories = vae.rnn.get_observation(Z, v=v, s=S, noise_scale=sim_obs_noise)
        
        if smooth: 
            window = signal.windows.hann(15) 
            data_gen = torch.from_numpy(
                zscore(
                    ndimage.convolve2d(data_gen.cpu().numpy(), window, axis=0), axis=0
                )
            )

        return trajectories.reshape(n_trials, -1, trial_dur)

def compute_r_2(signals_true, signals_pred):
    """
    Compute the R² (coefficient of determination) for a single neuron across multiple trials.
    
    Args: 
    signals_true (torch.Tensor; n_trials x time_steps): Actual signal
    signals_pred (torch.Tensor; n_trials x time_steps): Predicted signal 

    Returns:
        r_2 (float): the mean R² value across trials for the neuron 
        stdev (float): the stdev across trials for given neuron
    """
    mean_signals = signals_true.mean(axis=1)
    # compute variance of actual data 
    var = torch.sum((signals_true - mean_signals.unsqueeze(-1)) ** 2, axis=1)
    # compute residual sum of squares 
    ss_res = torch.sum((signals_true - signals_pred) ** 2, axis=1)
    r_2_arr = 1 - (ss_res / var)
    return r_2_arr.mean().item(), r_2_arr.std().item()


def compute_KL_divergence(pred_trajectories, x):
    """
    Computes the KL divergence over simulated and actual trajectories 
    
    Args: 
        pred_trajectories (torch.Tensor); n_trials x dim_x x time_steps: predicted trajectories 
        x (torch.Tensor): n_trials x dim_x x time_steps: actual neural activity 

    Returns: 
        kl_div: Estimated kernel divergence  
    """
    # reshape pred_trajectories and x 
    n_trials, trial_dur = pred_trajectories.shape[0], pred_trajectories.shape[2]
    pred_trajectories_ = pred_trajectories.T.reshape(n_trials*trial_dur, -1)
    x_ = x.T.reshape(n_trials*trial_dur, -1)
    kl_div = calc_kl_from_data(pred_trajectories_, x_)
    return kl_div.item()

def eval_VAE(
    vae,
    task,
    smoothing=20,
    cut_off=0,
    freq_cut_off=10000,
    sim_obs_noise=1,
    sim_latent_noise=1,
    smooth_at_eval=True,
    neuromodulation=False
):
    """
    Evaluate the VAE by looking at distribution over states and time

    Args:
        vae (nn.Module): VAE model
        task (Basic_dataset): dataset object
        smoothing (int): smoothing parameter for the power spectrum
        cut_off (int): cut off for the latent time series
        freq_cut_off (int): cut off for the power spectrum
        sim_obs_noise (float): observation noise scale
        sim_latent_noise (float): latent noise scale
        smooth_at_eval (bool): whether to smooth the generated data at evaluation time

    Returns:
        klx_bin (float): KL divergence between the true and generated data
        psH (float): power spectrum distance between the true and generated data
        mean_rate_error (float): mean rate error between the true and generated data

    """
    if neuromodulation:
        trial_data, _, _ = task.__getitem__(0)
    else:
        trial_data, _ = task.__getitem__(0)
        
    trial_dur = trial_data.shape[1]
    with torch.no_grad():

        # Evaluate on one long trajectory
        if len(task.data_eval.shape) == 2:
            data = task.data_eval
            T, dim_x = data.shape
            if sim_latent_noise > 1e-8:  # take sample of encoder
                z_hat, _, _, _ = vae.encoder(data[:trial_dur].T.unsqueeze(0))
            else:  # take mean prediction of encoder
                _, z_hat, _, _ = vae.encoder(data[:trial_dur].T.unsqueeze(0))
            z0 = z_hat[:, :, :1].squeeze()
            u, s = None, None 
            if task.task_input is not None: 
                u = torch.permute(task.task_input, 0, 2, 1)
            if task.s is not None: 
                s = torch.permute(task.s, 0, 2, 1)
            Z = vae.rnn.get_latent_time_series(
                time_steps=T, cut_off=cut_off, z0=z0, u=u, s=s, noise_scale=sim_latent_noise
            )
        # Evaluate on multiple short trajectories (trials)
        else:
            max_trials, T_data_trial, dim_x = task.data_eval.shape
            n_trials = int(10000 / T_data_trial)
            n_eval_trials = min(n_trials, max_trials)
            data = task.data_eval
            if sim_latent_noise > 1e-8:
                z_hat, _, _, _ = vae.encoder(data.permute(0, 2, 1))
            else:
                _, z_hat, _, _ = vae.encoder(data.permute(0, 2, 1))
            z0 = z_hat[:, :, :1]
            u, s = None, None 
            if task.task_input is not None: 
                u = torch.permute(task.task_input, (0, 2, 1))
            if task.s is not None: 
                s = torch.permute(task.s, (0, 2, 1))
            Z = vae.rnn.get_latent_time_series(
                time_steps=T_data_trial,
                cut_off=cut_off,
                z0=z0,
                u=u,
                s=s,
                noise_scale=sim_latent_noise,
            )

            data = data.reshape(dim_x, -1).T  # n_eval_trials*T_data_trial, dim_x
            T, dim_x = data.shape


        data_gen = (
            vae.rnn.get_observation(Z, noise_scale=sim_obs_noise)
            .permute(0, 2, 1, 3)
            .reshape(T, dim_x)
        )

        # potentially smooth
        if smooth_at_eval:
            window = signal.windows.hann(15)
            data_gen = torch.from_numpy(
                zscore(
                    ndimage.convolve1d(data_gen.cpu().numpy(), window, axis=0), axis=0
                )
            )

        klx_bin = calc_kl_from_data(data_gen, data.to(device=data_gen.device))

        # Helling distance accross time
        data = np.expand_dims(data.cpu().numpy(), 0)
        data_gen = np.expand_dims(data_gen.cpu().numpy(), 0)
        psH = power_spectrum_helling(
            data_gen, data, smoothing=smoothing, freq_cutoff=freq_cut_off
        )
        mean_rate_error = mean_rate(data_gen, data)
        print(
            f"KL_x = {klx_bin.item():.3f}, PS_dist = {psH:.3f}, Mean_rate_error = {mean_rate_error:.3f}"
        )
        return klx_bin.item(), psH, mean_rate_error


def mean_rate(data_gen, data_real):
    """Calculate the mean rate error between the true and generated data"""
    data_mean_rates = np.sum(data_real, axis=1)
    data_gen_mean_rates = np.sum(data_gen, axis=1)
    mean_rate_error = (
        np.mean((data_mean_rates - data_gen_mean_rates) ** 2) / data_gen.shape[1]
    )
    return mean_rate_error

def van_rossum_distance(spike_train1, spike_train2, tau, T):
    """
    Computes the Van Rossum distance between two spike trains.

    Args:
    - spike_train1: List or array of spikes of neuron 1.
    - spike_train2: List or array of spikes of neuron 2.
    - tau: Time constant for the exponential decay.
    - T: Total duration of the simulation (for defining the time axis).

    Returns:
    - Van Rossum distance between spike_train1 and spike_train2.
    """
    time_bins = np.arange(0, T, 1)

    # Define the exponential kernel
    kernel = np.exp(-time_bins / tau)
    kernel /= np.sum(kernel)  # Normalize kernel

    # Convolve spike trains with the exponential kernel
    filtered_spike_train1 = convolve(spike_train1, kernel, mode='same')
    filtered_spike_train2 = convolve(spike_train2, kernel, mode='same')

    # Compute the Euclidean distance between the convolved signals
    distance = np.sqrt(np.trapz((filtered_spike_train1 - filtered_spike_train2) ** 2, dx=1))


    return distance
