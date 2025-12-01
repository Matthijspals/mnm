import os 
import pandas as pd
import numpy as np
from scipy.signal import convolve, decimate
from scipy.signal.windows import gaussian

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
        spikes_df = pd.read_parquet(f'{spikes_dir}/{f}')
        spikes.append(spikes_df['signal'].to_numpy().flatten() / config['sampling_rate'])
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
        
    # load excitatory-inhibitory labels if available
    cell_types = None
    if config["dales_law"]:
        # load cell type of each neuron 
        cell_type_dir = os.path.join(config["data_dir"], 'labels')
        cell_types = pd.read_csv(f'{cell_type_dir}/neuron_labels.csv', header=None)

        # Convert cell types to 1 (Pyr) and -1 (Int)
        cell_types = cell_types[0].map({'Pyr': 1, 'Int': -1}).to_numpy()
        # Remove mask entries for low firing neurons that were removed
        cell_types = np.delete(cell_types, low_firing, axis=0)
        print(f'cell type: {cell_types}')

    # load neuromodulation signal 
    neuromod_dir = os.path.join(config['data_dir'], 'neuromodulators')
    if config['deconvolve']:
        lc_neuromod = pd.read_parquet(f'{neuromod_dir}/norepinephrine_deconvolved.parquet')['signal'].to_numpy().flatten()
    else:
        lc_neuromod = pd.read_parquet(f'{neuromod_dir}/norepinephrine_full.parquet')['signal'].to_numpy().flatten()
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

    # Normalize Neuromodulators
    if config['z_score_neuromod']:
        neuromod_activity = (lc_neuromod - lc_neuromod.mean()) / lc_neuromod.std()
    elif config['center_neuromod']:
        neuromod_activity += np.abs(neuromod_activity.min())
    else:
        neuromod_activity = lc_neuromod

    # mean-center the neurons 
    if config['center_data']:
        spike_counts = spike_counts - spike_counts.mean(axis=1, keepdims=True)
    if config['z_score_neurons']:
        spike_counts = spike_counts / spike_counts.std(axis=1, keepdims=True) 
    # spike_counts = (spike_counts - np.expand_dims(spike_counts.mean(axis=1), -1)) / np.expand_dims(spike_counts.std(axis=1), -1)
    print(np.isnan(spike_counts).any())
    if config['convolve_spikes']:
        kernel_size = 25
        sigma = 5

        # Create Gaussian kernel
        gaussian_kernel = gaussian(kernel_size, sigma)
        # Keep only the causal part (including the current time)
        #gaussian_kernel[:kernel_size // 2] = 0
        gaussian_kernel /= gaussian_kernel.sum()

        smoothed_spikes = []
        for n in range(spike_counts.shape[0]):
            if config['zero_pad']:
                # Zero-pad on the left so the first samples have enough past context
                original_length = spike_counts[n].shape[0]
                padded = np.pad(spike_counts[n], (kernel_size - 1, 0))
                # Convolve and take only the 'valid' part (no future context)
                smoothed = convolve(padded, gaussian_kernel, mode='valid')[:original_length]
                smoothed_spikes.append(smoothed)
            else: 
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

        return spikes, spike_counts, neuromod_activity, trials_df, whisking, pupil_area, cell_types 
        
    return spikes, spike_counts, neuromod_activity, trials_df, cell_types  


def get_block(spike_counts, neuromod_activity, t_start, trial_dur, bin_size, whisking=None, pupil_area=None):
    block_dur = int((trial_dur) / bin_size)
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

def prepare_dataset(spike_counts, neuromod_activity, trials_df, config, test=False, time_delta=0):
    """
    Creates blocks of baseline and trial activities and returns them for training / evaluation. 
    
    Args: 
        spike_counts (np.ndarray; neurons x time): Binned and smoothed spike counts for the entire dataset 
        neuromod_activity (np.ndarray; time): Smoothed neuromodulation signal 
        trials_df (pd.DataFrame): Metadata for trials 
        config (dict): Configuration parameters 
        test (bool): Train / Test flag 

    Returns: 
        x (np.ndarray; neurons x time): Horizontally concatenated blocks of population activity. 
                                        seq_periods array identifies where one block (for example baseline)
                                        starts and ends. 
        s (np.ndarray; time): Horizontally concatenated blocks of neuromodulation signal 
        stim_arr (np.ndarray; num_stimuli x time): Horizontally concatenated stimuli signals 
        seq_periods (list[list]): Identifier for where specific blocks of activity begin and end 
        white_noise_trial_avgs (np.ndarray; neurons x time): Trial averaged population activity for evaluation 
        airpuff_trial_avgs (np.ndarray; neurons x time): Trial averaged population activity for evaluation 
        reward_trial_avgs (np.ndarray; neurons x time): Trial averaged population activity for rewards 
        time_delta (int): number of seconds before and after trial to append to data for model to learn transitions to/from stimuli regimes
    """ 
    # define training samples
    x = None 
    s = None 
    stim_arr = None

    num_stimuli = 9
    bin_size = config["bin_size"]

    white_noise_trial_dur = 6 # 6 seconds
    airpuff_trial_dur = 13.5
    reward_trial_dur = 215 

    if test:
        spike_counts_baseline, s_baseline = get_block(spike_counts, 
                                                      neuromod_activity, 
                                                      config["baseline_test_start"], 
                                                      config["baseline_test_end"] - config["baseline_test_start"], 
                                                      bin_size)
    else:
        spike_counts_baseline, s_baseline = get_block(spike_counts, 
                                                      neuromod_activity, 
                                                      config["baseline_train_start"], 
                                                      config["baseline_train_end"]- config["baseline_train_start"],
                                                      bin_size)

    x = [spike_counts_baseline]
    s = [s_baseline] 
    stim_arr = [np.zeros((num_stimuli, s_baseline.shape[0]))]
    seq_periods = [[0, s_baseline.shape[0]]]

    #TODO: organize this part 
    # There are 8 white noise blocks. First 4 for training and second 4 for testing  

    white_noise_trials_x = []
    white_noise_trials_s = [] 
    white_noise_trials_stim = []

    white_noise_trials_cs = trials_df[trials_df['trialtype'] == 'TONE_2']['cs'].to_list()

    zero_arr = np.zeros((num_stimuli, int(time_delta / bin_size))) 

    for trial_idx, trial in enumerate(white_noise_trials_cs):
        spike_counts_white_noise, s_white_noise = get_block(spike_counts, neuromod_activity, trial - time_delta, white_noise_trial_dur + time_delta * 2, bin_size)
        # get stimulus 
        stim_white_noise_trial = get_white_noise_trial(trial, bin_size) 
        white_noise_trials_x.append(spike_counts_white_noise)
        white_noise_trials_s.append(s_white_noise) 
        white_noise_trials_stim.append(np.hstack([zero_arr, stim_white_noise_trial, zero_arr])) 
        
        if (test and trial_idx >= 4) or (test == False and trial_idx < 4):
            seq_periods.append([seq_periods[-1][1], seq_periods[-1][1] + s_white_noise.shape[0]])

    white_noise_trial_avgs = np.stack(white_noise_trials_x).mean(axis=0)
    if test:
        x += white_noise_trials_x[4:]
        s += white_noise_trials_s[4:]
        stim_arr += white_noise_trials_stim[4:]
    else:
        x += white_noise_trials_x[:4]
        s += white_noise_trials_s[:4]
        stim_arr += white_noise_trials_stim[:4]

    #TODO: organize this part 
    # There are 6 airpuff blocks. get first 3 for training and use remaining for testing  
    # We also need the trial averages for evaluation, so we will retrieve all trials first
    
    airpuff_trials_x = []
    airpuff_trials_s = [] 
    airpuff_trials_stim = [] 

    airpuff_trial_cs = trials_df[trials_df['trialtype'] == 'TONE_3_LAG_500_AIRPUFF_1']['cs'].to_list()
    airpuff_trial_us = trials_df[trials_df['trialtype'] == 'TONE_3_LAG_500_AIRPUFF_1']['us'].to_list()
    
    for trial_idx, trial in enumerate(airpuff_trial_cs):
        spike_counts_airpuff, s_airpuff = get_block(spike_counts, neuromod_activity, trial - time_delta, airpuff_trial_dur + time_delta * 2, bin_size)
        us = airpuff_trial_us[trial_idx]
        stim_airpuff = get_airpuff_trial(trial, us, bin_size) 

        airpuff_trials_x.append(spike_counts_airpuff)
        airpuff_trials_s.append(s_airpuff) 
        airpuff_trials_stim.append(np.hstack([zero_arr, stim_airpuff, zero_arr]))
 
        if (test and trial_idx >= 3) or (test == False and trial_idx < 3):
            seq_periods.append([seq_periods[-1][1], seq_periods[-1][1] + s_airpuff.shape[0]])

    airpuff_trial_avgs = np.stack(airpuff_trials_x).mean(axis=0)
    if test:
        x += airpuff_trials_x[3:]
        s += airpuff_trials_s[3:]
        stim_arr += airpuff_trials_stim[3:]
    else:
        x += airpuff_trials_x[:3]
        s += airpuff_trials_s[:3]
        stim_arr += airpuff_trials_stim[:3]

    # There are 7 reward blocks. First 4 go for training and remaining 3 for testing
    reward_trials_x = []
    reward_trials_s = [] 
    reward_trials_stim = []

    for i in np.arange(0, 7):
        # get stimuli 
        stim_arr_rewards, trial_blocks, reward_trial_start = get_reward_block(i, bin_size, trials_df)
        spike_counts_reward, s_reward = get_block(spike_counts, neuromod_activity, reward_trial_start - time_delta, reward_trial_dur + 2 * time_delta, bin_size)
        reward_trials_x.append(spike_counts_reward)
        reward_trials_s.append(s_reward)
        reward_trials_stim.append(np.hstack([zero_arr, stim_arr_rewards, zero_arr]))
        
        if (test and i >= 4) or (test == False and i < 4):
            seq_periods.append([seq_periods[-1][1], seq_periods[-1][1] + s_reward.shape[0]])

    reward_trial_avgs = np.stack(reward_trials_x).mean(axis=0)
    if test:
        x += reward_trials_x[4:]
        s += reward_trials_s[4:]
        stim_arr += reward_trials_stim[4:]
    else:
        x += reward_trials_x[:4]
        s += reward_trials_s[:4]
        stim_arr += reward_trials_stim[:4]
    print(len(x))

    x = np.hstack(x)
    s = np.hstack(s) 

    print(s_baseline.shape, s.shape)
    stim_arr = np.hstack(stim_arr) 

    if config["shuffle"]:
         print('Randomly shuffling Neuromodulators')
         np.random.shuffle(s)
    print(f'Shape of data samples: {x.shape}, {s.shape}')
    return x, s, stim_arr, seq_periods, white_noise_trial_avgs, airpuff_trial_avgs, reward_trial_avgs