import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import torch
import scipy.ndimage as ndimage
import numpy as np
from vi_rnn.generate import generate, get_initial_state
from kl_Gauss_np import calc_kl_from_data
from pse import power_spectrum_helling

from scipy.signal import convolve
from scipy.signal.windows import gaussian 

def eval_pairwise_corr(x, x_gen, mask=None, verbose=True, eps = 1e-6):
    """
    Efficiently compute pairwise correlations between neurons in x and x_gen.
    
    Args:
        x: array (trials, N, timesteps) original data
        x_gen: array (trials, N, timesteps) generated data
        mask: array (trials, timesteps) boolean mask, if None all time points are used
        
    Returns:
        corr: array (N, N) pairwise correlations of x
        corr_gen: array (N, N) pairwise correlations of x_gen
        r2: float, R^2 between off-diagonal correlations of x and x_gen
    """
    # Flatten over trials and time
    n_units = x.shape[1]
    x_flat = x.transpose(0,2,1).reshape(-1, n_units)
    x_gen_flat = x_gen.transpose(0,2,1).reshape(-1, n_units)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        x_flat = x_flat[mask_flat]
        x_gen_flat = x_gen_flat[mask_flat]
    var_x = x_flat.var(axis=0)
    var_gen = x_gen_flat.var(axis=0)

    # A neuron is "bad" if EITHER x or x_gen is constant
    bad = (var_x < eps) | (var_gen < eps)
    n_bad = bad.sum()

    if verbose and n_bad > 0:
        print(f"Removing {n_bad} constant-variance neurons out of {n_units} total.")

    # Keep only good neurons
    good = ~bad
    x_flat = x_flat[:, good]
    x_gen_flat = x_gen_flat[:, good]


    # Compute correlation matrices (vectorized)
    corr = np.corrcoef(x_flat, rowvar=False)
    corr_gen = np.corrcoef(x_gen_flat, rowvar=False)

    # Compute R^2 of off-diagonal elements
    offdiag = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    r2 = np.corrcoef(corr[offdiag], corr_gen[offdiag])[0, 1] ** 2
    if verbose:
        print(f"R^2 of pairwise correlations: {r2:.3f}")
    return corr[offdiag], corr_gen[offdiag], r2


def eval_spikestats(x_test, vae=None, s_test = None, u_test = None, x_gen = None, initial_state="prior_sample", verbose=True, num_samples = 3000,
             min_data_points_isi = 5, return_raw_data=False, dt=1, smooth_spikes_KL=True, smooth_sigma_KL=5, pse_smooth=20, pse_freq_cutoff=200,
             data_mean=0, eval_spikes=True
             ):

    """
    Evaluate spiking statistics between data and generated data from VAE
    Either pass in generated data x_gen, or pass in vae tigether with u_test and s_test to generate data
    
    """

    data_dict = {"mean_rate":0,
                "mean_ISI":0,
                "std_ISI":0,
                "r2_pwcorr":0,
                "KL_data":np.inf,
                "power_spectr_distance":1,
                }
    raw_data_dict = {"mean_rate":[] ,
                "mean_ISI":[],
                "std_ISI":[],
                "pwcorr":[],
                "KL_data":[],
                "power_spectr_distance":[],
                }
 
    kernel_size = smooth_sigma_KL*5 
    total_gen = num_samples + 2 * kernel_size
    # add trial dim if necessary
    if x_test.dim() == 2:
        x_test = x_test.unsqueeze(0)
    x_test = x_test[:, :, :total_gen]

    if x_gen is None:
        print("Generating data from VAE for evaluation...")
        assert vae is not None, "Either x_gen or vae must be provided"
        assert s_test is not None, "s_test must be provided when vae is provided"
        assert u_test is not None, "u_test must be provided when vae is provided"
        # add trial dimension and limit to num_samples
        u_test = u_test[:,:, :total_gen]
        s_test = s_test[:,:, :total_gen]


        print("STATISTICS FOR GENERATION")
        print("shapes:", u_test.shape, s_test.shape, x_test.shape)
        print("s_test stats: mean", s_test.mean().item(), "std", s_test.std().item())
        print("x_test stats: mean", x_test.mean().item(), "std", x_test.std().item())
        print("u_test stats: mean", u_test.mean().item(), "std", u_test.std().item())

        _, _, data_gen_test,_ = generate(
            vae,
            u=u_test,
            s=s_test,
            x=x_test[:,:,:50], # only for getting the initial state
            initial_state=initial_state,
            k=1,
        )
        # check if nan in generated data
        if torch.isnan(data_gen_test).any():
            print("NaN in generated data!")
            return data_dict
            #raise ValueError("NaN in generated data!")

        data_gen_test = data_gen_test.cpu().numpy()[:,:,:,0]
    else:
        # add trial dim if necessary
        if x_gen.dim() == 2:
            x_gen = x_gen.unsqueeze(0)
        data_gen_test = x_gen[:, :, :total_gen].cpu().numpy()
    
    x_test = x_test.cpu().numpy()
    #print(x_test.shape, data_gen_test.shape)
    #(1, 184, 5760) (1, 184, 5760) # trials, units, timesteps

    n_trials, n_units, T = x_test.shape

    # Compute total spikes and total time per unit (collapsed)

    # sum over trials and time per unit
    total_spikes     = x_test.sum(axis=(0, 2))
    total_spikes_gen = data_gen_test.sum(axis=(0, 2))

    # total time per unit (same for all units)
    n_trials, _, n_time = x_test.shape
    total_time = n_trials * n_time * dt

    mean_rates     = total_spikes     / total_time   # Hz
    mean_rates_gen = total_spikes_gen / total_time   # Hz

    data_dict["mean_rate"] += np.corrcoef(mean_rates, mean_rates_gen)[0, 1]
    raw_data_dict["mean_rate"].append([mean_rates, mean_rates_gen])

    


    # ISI distributions --------------------------------

    mean_isis     = np.zeros(n_units)
    mean_isis_gen = np.zeros(n_units)
    std_isis      = np.zeros(n_units)
    std_isis_gen  = np.zeros(n_units)

    for ni in range(n_units):
        isi_all     = []
        isi_gen_all = []
        
        # slice once: (trials, time)
        unit_spikes = x_test[:, ni, :]
        gen_spikes  = data_gen_test[:, ni, :]
        
        for ti in range(n_trials):
            spk_real = unit_spikes[ti]
            spk_gen  = gen_spikes[ti]
            
            idx_real = np.flatnonzero(spk_real)
            if idx_real.size > 1:
                isi_all.append(np.diff(idx_real))
            
            idx_gen = np.flatnonzero(spk_gen)
            if idx_gen.size > 1:
                isi_gen_all.append(np.diff(idx_gen))
        
        # Evaluate after accumulating
        if isi_all and sum(map(len, isi_all)) > min_data_points_isi:
            isi_cat = np.concatenate(isi_all) * dt
            mean_isis[ni] = isi_cat.mean()
            std_isis[ni]  = isi_cat.std()
        
        if isi_gen_all and sum(map(len, isi_gen_all)) > min_data_points_isi:
            isi_cat = np.concatenate(isi_gen_all) * dt
            mean_isis_gen[ni] = isi_cat.mean()
            std_isis_gen[ni]  = isi_cat.std()


    # Compute correlations -----------------------------
    valid = (
        (mean_isis > 0) &
        (mean_isis_gen > 0) &
        (std_isis > 0) &
        (std_isis_gen > 0)
    )   

    if verbose:
        print(f"ignoring {np.sum(~valid)} units for ISI baseline calculation")
    #print( std_isis_gen[valid])
    #print(mean_isis_gen[valid])
    if sum(~valid)>len(valid)//2:
        if verbose:
            print("Warning: more than half of units invalid for ISI calculation, setting rs to zero")
    elif np.std(mean_isis[valid])<1e-6 or np.std(mean_isis_gen[valid])<1e-6:
        if verbose:
            print("Warning: zero variance in mean ISI, setting r to zero")
    else:
        cf_mean_isi = np.corrcoef(mean_isis[valid], mean_isis_gen[valid])[0, 1]
        cf_std_isi  = np.corrcoef(std_isis[valid],  std_isis_gen[valid])[0, 1]
        data_dict["mean_ISI"] = cf_mean_isi
        data_dict["std_ISI"]  = cf_std_isi

        raw_data_dict["mean_ISI"].append([mean_isis[valid], mean_isis_gen[valid]])
        raw_data_dict["std_ISI"].append([std_isis[valid], std_isis_gen[valid]])

    
    # Calculate correlation matrices
    #---------------------------------
    m_test = None
    corr, corr_gen, r2 = eval_pairwise_corr(x_test, data_gen_test,m_test, verbose=verbose)

    data_dict["r2_pwcorr"] = r2
    raw_data_dict["pwcorr"].append([corr, corr_gen])

    # KL and PSE
    # ---------------------------------

    # convolve and then mean-center the spikes for comparison 
    valid_start = kernel_size
    valid_end   = valid_start + num_samples
    if smooth_spikes_KL:
        sigma = smooth_sigma_KL

        gaussian_kernel = gaussian(kernel_size, sigma) 
        gaussian_kernel /= gaussian_kernel.sum() 
        n_trials = x_test.shape[0]
        smoothed_spikes, smoothed_spikes_gen = [[] for _ in range(n_trials)], [[] for _ in range(n_trials)]
        for tr in range(n_trials):
            for n in range(x_test.shape[1]): 
                smoothed_spikes[tr].append(convolve(x_test[tr, n, :], gaussian_kernel, mode='full'))
                smoothed_spikes_gen[tr].append(convolve(data_gen_test[tr, n, :], gaussian_kernel, mode='full'))
        
        smoothed_spikes = np.array(smoothed_spikes)
        smoothed_spikes_gen = np.array(smoothed_spikes_gen)
        print("After smoothing, shapes:", smoothed_spikes.shape, smoothed_spikes_gen.shape)
        print(smoothed_spikes.shape, data_mean.shape)
        smoothed_spikes = smoothed_spikes - data_mean#smoothed_spikes.mean(axis=2, keepdims=True)
        smoothed_spikes_gen = smoothed_spikes_gen - data_mean#smoothed_spikes_gen.mean(axis=2, keepdims=True) 

        # remove kernel convolution effects
        smoothed_spikes = smoothed_spikes[:, :, valid_start:valid_end]
        smoothed_spikes_gen = smoothed_spikes_gen[:, :, valid_start:valid_end]

    else:
        smoothed_spikes = x_test - data_mean#x_test.mean(axis=2, keepdims=True)
        smoothed_spikes_gen = data_gen_test - data_mean#data_gen_test.mean(axis=2, keepdims=True)
        smoothed_spikes = smoothed_spikes[:, :, valid_start:valid_end]
        smoothed_spikes_gen = smoothed_spikes_gen[:, :, valid_start:valid_end]
    
    # Helling distance accross time, doesn't seem to be reliable 
    # some measure of distributional distance between power spectra
    # would be good though...
    psH = power_spectrum_helling(smoothed_spikes_gen.transpose(0, 2, 1), smoothed_spikes.transpose(0, 2, 1), smoothing=pse_smooth, freq_cutoff=pse_freq_cutoff)
    data_dict["power_spectr_distance"] = psH
    raw_data_dict["power_spectr_distance"].append(psH)
     # first flatten data and mask

    x_test = smoothed_spikes.transpose(0, 2, 1).reshape(-1, n_units)
    data_gen_test = smoothed_spikes_gen.transpose(0, 2, 1).reshape(-1, n_units)

    # KL accross states
    klx_bin = calc_kl_from_data(data_gen_test, x_test)

    data_dict["KL_data"] = klx_bin
    raw_data_dict["KL_data"].append(klx_bin)

    #print all data:
    if verbose:
        print("Evaluation results:")
        print("---------------")
        for key in data_dict.keys():
            print(key + " " + str(data_dict[key]))
        print("---------------")

    if return_raw_data:
        return data_dict, raw_data_dict
    
    return data_dict
