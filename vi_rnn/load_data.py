import os 
import pandas as pd
import numpy as np
from scipy.signal import convolve, gaussian, decimate

def bin_spike_train(spike_times, t_start, t_end, bin_size):
    """
    Bin the spike train given spike times 

    Args:
        spike_times (numpy.ndarray; num_spikes x 1): Times (in ms) in which spikes occur
        t_start (int): start time
        t_end (int): end time
        bin_size (int): bin width in ms

    Returns: 
        binned_counts (numpy.ndarray; num_bins x 1): Number of spikes in each bin
        bin_edges (numpy.ndarray; num_bins x 1): Time edges of the bins   
    """
    # exclude spikes after t_end 
    spike_times = spike_times[spike_times <= t_end]
    bins = np.arange(t_start, t_end + bin_size, bin_size)
    binned_counts, bin_edges = np.histogram(spike_times, bins=bins)
    return binned_counts, bin_edges 


def spike_times_to_binary(spike_times, t_start, t_end):
    """
    Convert from neuron spike times to a binary array for each time point

    Args:
        spike_times (list[numpy.ndarray]; num_neurons x num_spikes)
        t_start (int): start time 
        t_end (int): end time 
    """
    bin_arr = np.zeros((len(spike_times), (t_end - t_start + 1)))
    for i in range(len(spike_times)):
        spike_times[i] = spike_times[i][spike_times[i] <= t_end]
        bin_arr[i, int(spike_times[i])] = 1.0
    return bin_arr


def load_data(config):
    spikes = []

    spikes_dir = os.path.join(config['data_dir'], 'electrophysiology')
    files = [f for f in os.listdir(spikes_dir) if os.path.isfile(os.path.join(spikes_dir, f))]
    files = sorted(files)
    t_start, t_end = 0, 0
    for f in files:
        spikes_df = pd.read_csv(f'{spikes_dir}/{f}', header=None)
        spikes.append(spikes_df.to_numpy().flatten() / config['sampling_rate'])
        t_end = max(t_end, int(np.ceil(spikes[-1].max())))
    
    # remove twenty seconds in the end  
    t_end = t_end - 20
    print(f't_end = {t_end}')
    spike_counts = np.zeros((len(spikes), int((t_end - t_start) / config['bin_size'])))
    for i in range(len(spikes)):
        spike_counts[i], _ = bin_spike_train(spikes[i], t_start, t_end, config['bin_size'])
        
    if config['threshold_neurons']:
        rates = spike_counts.mean(axis=1) / config['bin_size']
        low_firing = np.where(rates < config['fr_threshold'])[0] 
        print(f'{len(low_firing)} neurons removed') 
        spike_counts = np.delete(spike_counts, low_firing, axis=0)
        
    # load neuromodulation signal 
    neuromod_dir = os.path.join(config['data_dir'], 'neuromodulators')
    if config['deconvolve']:
        lc_neuromod = pd.read_csv(f'{neuromod_dir}/norepinephrine_deconvolved_2.csv', header=None).to_numpy().flatten()
    else:
        lc_neuromod = pd.read_csv(f'{neuromod_dir}/norepinephrine_full.csv', header=None).to_numpy().flatten()
    # downsample
    fs, target_fs = 1000, 1 / config['bin_size']
    downsample_factor = int(fs / target_fs)
    downsampled_signal = decimate(lc_neuromod, downsample_factor, ftype='fir', zero_phase=True)
    print(f'length of downsampled signal: {len(downsampled_signal)}')
    print(f'length of original signal: {len(lc_neuromod)}') 
    lc_neuromod = downsampled_signal 
    
    # select in window 
    lc_neuromod = lc_neuromod[int(t_start / config['bin_size']):int(t_end / config['bin_size'])]
    # replace nans with zero 
    lc_neuromod = np.nan_to_num(lc_neuromod)
    # z-score 
    if config['z_score']:
        neuromod_activity = (lc_neuromod - lc_neuromod.mean()) / lc_neuromod.std()
    else:
        neuromod_activity = lc_neuromod
    # center 
    if config['center']:
        neuromod_activity += np.abs(neuromod_activity.min())
    # mean-center the neurons 
    spike_counts = spike_counts - spike_counts.mean(axis=1, keepdims=True)
    # spike_counts = (spike_counts - np.expand_dims(spike_counts.mean(axis=1), -1)) / np.expand_dims(spike_counts.std(axis=1), -1)
    print(np.isnan(spike_counts).any())
    if config['convolve_spikes']:
        kernel_size = 25
        sigma = 5
        
        gaussian_kernel = gaussian(kernel_size, sigma) 
        # Make it causal: Zero out future values
        # gaussian_kernel[:kernel_size // 2] = 0
        gaussian_kernel /= gaussian_kernel.sum()
        smoothed_spikes = []
        for n in range(spike_counts.shape[0]):
            smoothed_spikes.append(convolve(spike_counts[n], gaussian_kernel, mode='full'))
        spike_counts = np.array(smoothed_spikes)
    # load trials info 
    print(np.isnan(spike_counts).any())
    trials_dir = os.path.join(config['data_dir'], 'trials') 
    trials_df = pd.read_csv(os.path.join(trials_dir, 'trials.csv'))

    if config['load_physiology']:
        phys_dir = os.path.join(config['data_dir'], 'physiology')
        whisking = pd.read_csv(os.path.join(phys_dir, 'whisking.csv')).to_numpy().flatten()
        pupil_area = pd.read_csv(os.path.join(phys_dir, 'pupil_area.csv')).to_numpy().flatten()
        # wheel = pd.read_csv(os.path.join(phys_dir, 'wheel.csv')).to_numpy().flatten()

        # downsample all signals 
        whisking = decimate(whisking, downsample_factor, ftype='fir', zero_phase=True)
        pupil_area = decimate(pupil_area, downsample_factor, ftype='fir', zero_phase=True)
        # wheel = decimate(wheel, downsample_factor, ftype='fir', zero_phase=True)

        # select in window
        whisking = whisking[int(t_start / config['bin_size']):int(t_end / config['bin_size'])]
        pupil_area = pupil_area[int(t_start / config['bin_size']):int(t_end / config['bin_size'])]
        # wheel = wheel[int(t_start / config['bin_size']):int(t_end / config['bin_size'])]

        whisking = np.nan_to_num(whisking)
        pupil_area = np.nan_to_num(pupil_area)
        # wheel = np.nan_to_num(wheel)

        return spikes, spike_counts, neuromod_activity, trials_df, whisking, pupil_area 
        
    return spikes, spike_counts, neuromod_activity, trials_df 


def get_block(spike_counts, neuromod_activity, t_start, t_end, bin_size, whisking=None, pupil_area=None):
    block_dur = int((t_end - t_start) / bin_size)
    block_start_idx = int(t_start / bin_size) 
    spike_counts_block = spike_counts[:, block_start_idx:block_start_idx + block_dur]
    s_block = neuromod_activity[block_start_idx:block_start_idx + block_dur]
    
    if whisking is not None and pupil_area is not None: 
        whisking_block = neuromod_activity[block_start_idx:block_start_idx + block_dur] 
        pupil_area_block = pupil_area[block_start_idx:block_start_idx + block_dur]
        return spike_counts_block, s_block, whisking_block, pupil_area_block
        
    return spike_counts_block, s_block 

def get_trial_blocks(trials_df, trial_type):
    trial_times = trials_df[trials_df['trialtype'] == trial_type]['t'].astype('int').to_list()
    trial_blocks = []
    for t in trial_times:
        t_start, t_end = t - 10, t + 10
        trial_blocks.append([t_start, t_end])
    return trial_blocks

def get_white_noise_trial(cs, bin_size):
    trial_dur = 6
    stim_arr_white_noise = np.zeros((9, int(trial_dur / bin_size),))
    trial_start = cs 
    pip_dur = 100 
    pip_interval = 200 
    stim_idx = 0
    for i in range(20):
        stim_arr_white_noise[3, int(stim_idx + pip_dur / (bin_size * 1000))] = 1.0 
        stim_idx += int(pip_dur / (bin_size * 1000)) + int(pip_interval / (bin_size * 1000))
    return stim_arr_white_noise

def get_airpuff_trial(cs, us, bin_size):
    trial_dur = 13.5 # 13.5 seconds 
    tone_dur = 100 # 100 ms  
    airpuff_dur = 100 # 100 ms 
    airpuff_interval = 100 # 100 ms 
    tone_interval = 900 
    
    stim_arr_airpuff = np.zeros((9, int(trial_dur / bin_size),))
    stim_idx = 0
    for i in range(10):
        stim_arr_airpuff[2, int(stim_idx):int(stim_idx + tone_dur / (bin_size * 1000))] = 1.0
        stim_idx += int(tone_dur / (bin_size * 1000)) + int(tone_interval / (bin_size * 1000))
    # now add the unconditioned stimulus 
    stim_idx = (us - cs) / bin_size 
    for i in range(15):
        stim_arr_airpuff[4, int(stim_idx):int(stim_idx + airpuff_dur / (bin_size * 1000))] = 1.0 
        stim_idx += int(airpuff_dur / (bin_size * 1000)) + int(airpuff_interval / (bin_size * 1000))
    return stim_arr_airpuff

def get_reward_block(block_idx, bin_size, trials_df):
    block_start_idx = 7 * block_idx 
    trial_dur = 215 # 215s
    tone_dur = 1000 # 1000 ms
    lag_dur = 500 # 500 ms
    reward_dur = 2000 # 2000

    stim_arr_reward = np.zeros((9, int(trial_dur / bin_size),))
    trial_blocks = []
    # Add 4 trials of reward 1
    reward_1_us = trials_df[trials_df['trialtype'] == 'TONE_1_LAG_500_REW_1_LAG_2000_VACUUM_1']['cs'].to_list()[block_start_idx:block_start_idx + 7]
    for i in range(4):
        tone_1_onset = reward_1_us[i]
        trial_start = (tone_1_onset - reward_1_us[0]) // bin_size
        stim_idx = int(trial_start)
        # Add 1s tone 
        stim_arr_reward[0, stim_idx:int(stim_idx + tone_dur / (bin_size * 1000))] = 1.0
        stim_idx += int(tone_dur / (bin_size * 1000)) + int(lag_dur / (bin_size * 1000))
        # Provide reward 
        stim_arr_reward[5, stim_idx:int(stim_idx + reward_dur / (bin_size * 1000))] = 1.0
        trial_blocks.append([trial_start, int(stim_idx + reward_dur / (bin_size * 1000))])
        
    # omission (tone without reward)
    omission_onset = trials_df[trials_df['trialtype'] == 'TONE_1_LAG_2500']['cs'].to_list()[block_idx]
    trial_start = (omission_onset - reward_1_us[0]) / bin_size 
    stim_idx = int(trial_start) 
    stim_arr_reward[0, stim_idx:int(stim_idx + tone_dur / (bin_size * 1000))] = 1.0
    trial_blocks.append([trial_start, int(stim_idx + tone_dur / (bin_size * 1000))])

    # reward 2 
    reward_2_us = trials_df[trials_df['trialtype'] == 'TONE_1_LAG_500_REW_2_LAG_2000_VACUUM_1']['cs'].to_list()[block_idx*3:block_idx*3 +3]
    for i in range(3):
        trial_start = (reward_2_us[i] - reward_1_us[0]) // bin_size 
        stim_idx = int(trial_start)
        stim_arr_reward[0, stim_idx:int(stim_idx + tone_dur / (bin_size * 1000))] = 1.0
        stim_idx += int(tone_dur / (bin_size * 1000)) + int(lag_dur / (bin_size * 1000))
        # provide 10% sucrose drop 
        stim_arr_reward[6, stim_idx:int(stim_idx + reward_dur / (bin_size * 1000))] = 1.0 
        stim_idx += reward_dur / (bin_size * 1000)
        trial_blocks.append([trial_start, int(stim_idx)])
        
    # quinine 
    quinine_onset = trials_df[trials_df['trialtype'] == 'TONE_1_LAG_500_REW_3_LAG_2000_VACUUM_1']['cs'].to_list()[block_idx]
    trial_start = (quinine_onset - reward_1_us[0]) / bin_size
    stim_idx = int(trial_start)
    stim_arr_reward[0, stim_idx:int(stim_idx + tone_dur / (bin_size * 1000))] = 1.0
    stim_idx += int(tone_dur / (bin_size * 1000)) + int(lag_dur / (bin_size * 1000))
    # provide quinine drop 
    stim_arr_reward[7, stim_idx:int(stim_idx + reward_dur / (bin_size * 1000))] = 1.0    
    stim_idx += int(reward_dur / (bin_size * 1000))
    trial_blocks.append([trial_start, stim_idx])

    # reward 1 again 
    for i in range(4, 7):
        tone_1_onset = reward_1_us[i]
        trial_start = (tone_1_onset - reward_1_us[0]) // bin_size
        stim_idx = int(trial_start)
        # Add 1s tone 
        stim_arr_reward[0, stim_idx:int(stim_idx + tone_dur / (bin_size * 1000))] = 1.0
        stim_idx += int(tone_dur / (bin_size * 1000)) + int(lag_dur / (bin_size * 1000))
        # Provide reward 
        stim_arr_reward[5, stim_idx:int(stim_idx + reward_dur / (bin_size * 1000))] = 1.0
        trial_blocks.append([trial_start, int(stim_idx + reward_dur / (bin_size * 1000))])
    return stim_arr_reward, trial_blocks, reward_1_us[0]


