import os 
import sys
import argparse 
sys.path.append("../../../")

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader, Dataset

from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.load_data import *
from vi_rnn.utils import *
from vi_rnn.evaluation import * 
from vi_rnn.saving import save_model, load_model
from vi_rnn.datasets import DTTDataset

from py_rnn.model import RNN, predict
from py_rnn.train import train_rnn
from py_rnn.train import save_rnn, load_rnn
from py_rnn.default_params import get_default_params

import pandas as pd 

from sklearn.model_selection import train_test_split, KFold

from evaluation.calc_stats import calc_isi_stats
from scipy.signal import convolve, gaussian, decimate


def load_data(config):
    spikes = []

    spikes_dir = os.path.join(config["data_dir"], "electrophysiology")
    files = [f for f in os.listdir(spikes_dir) if os.path.isfile(os.path.join(spikes_dir, f))]
    files = sorted(files)
    t_start, t_end = 0, 0
    for f in files:
        spikes_df = pd.read_csv(f"{spikes_dir}/{f}", header=None)
        spikes.append(spikes_df.to_numpy().flatten() / config["sampling_rate"])
        t_end = max(t_end, int(np.ceil(spikes[-1].max())))
    
    # remove twenty seconds in the end  
    t_end = t_end - 20
    print(f"t_end = {t_end}")
    spike_counts = np.zeros((len(spikes), int((t_end - t_start) / config["bin_size"])))
    for i in range(len(spikes)):
        spike_counts[i], _ = bin_spike_train(spikes[i], t_start, t_end, config["bin_size"])
    rates = spike_counts.mean(axis=1) / config["bin_size"] 
    low_firing = np.where(rates < config["fr_threshold"])[0] 
    print(f'{len(low_firing)} neurons removed') 
    spike_counts = np.delete(spike_counts, low_firing, axis=0)
    
    # load neuromodulation signal 
    neuromod_dir = os.path.join(config["data_dir"], 'neuromodulators')
    lc_neuromod = pd.read_csv(f'{neuromod_dir}/norepinephrine_full.csv', header=None).to_numpy().flatten()
    # downsample
    fs, target_fs = 1000, 1 / config["bin_size"]
    downsample_factor = int(fs / target_fs)
    downsampled_signal = decimate(lc_neuromod, downsample_factor, ftype='fir', zero_phase=True)
    print(f'length of downsampled signal: {len(downsampled_signal)}')
    print(f'length of original signal: {len(lc_neuromod)}') 
    lc_neuromod = downsampled_signal 
    
    # select in window 
    lc_neuromod = lc_neuromod[int(t_start/config["bin_size"]):int(t_end/config["bin_size"])]
    # replace nans with zero 
    lc_neuromod = np.nan_to_num(lc_neuromod)
    # z-score 
    if config["z_score"]:
        neuromod_activity = (lc_neuromod - lc_neuromod.mean()) / lc_neuromod.std()
    else:
        neuromod_activity = lc_neuromod
    
    # mean-center the neurons 
    spike_counts = spike_counts - spike_counts.mean(axis=1, keepdims=True)
    
    if config["convolve_spikes"]:
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
    trials_dir = os.path.join(config["data_dir"], 'trials') 
    trials_df = pd.read_csv(os.path.join(trials_dir, 'trials.csv'))

    return spikes, spike_counts, neuromod_activity, trials_df 

def get_block(spike_counts, neuromod_activity, t_start, t_end, bin_size):
    block_dur = int((t_end - t_start) / bin_size)
    block_start_idx = int(t_start / bin_size) 
    spike_counts_block = spike_counts[:, block_start_idx:block_start_idx + block_dur]
    s_block = neuromod_activity[block_start_idx:block_start_idx + block_dur]
    return spike_counts_block, s_block 

def get_trial_blocks(trial_type):
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

def get_reward_block(block_idx, bin_size):
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


def prepare_dataset(spike_counts, neuromod_activity, trials_df, config):
    # define training samples
    x_train = None 
    s_train = None 
    stim_arr_train = None 
    bin_size = config["bin_size"]
    
    spike_counts_pt_baseline, s_pt_baseline = get_block(spike_counts, neuromod_activity, config["t_start_post_trial_train"], config["t_end_post_trial_train"], bin_size)
    # define normalization parameters
    s_min, s_max = s_pt_baseline.min(), s_pt_baseline.max() 

    x_train = [spike_counts_pt_baseline]
    s_train = [s_pt_baseline] 
    stim_arr_train = [np.zeros((9, s_pt_baseline.shape[0]))]
    seq_periods = [[0, s_pt_baseline.shape[0]]]
    
    # get first 4 white noise blocks 
    white_noise_trial_dur = 6 # 6 seconds
    white_noise_trials_cs = trials_df[trials_df['trialtype'] == 'TONE_2']['cs'].to_list()[0:4] 
    zero_arr = np.zeros((9, int(15 / bin_size))) 

    for trial_idx, trial in enumerate(white_noise_trials_cs):
        spike_counts_white_noise, s_white_noise = get_block(spike_counts, neuromod_activity, trial - 15, trial + white_noise_trial_dur + 15, bin_size)
        s_min = np.minimum(s_min, s_white_noise.min())
        s_max = np.maximum(s_max, s_white_noise.max())
        # get stimulus 
        stim_white_noise_trial = get_white_noise_trial(trial, bin_size) 
        x_train.append(spike_counts_white_noise)
        s_train.append(s_white_noise) 
        stim_arr_train.append(np.hstack([zero_arr, stim_white_noise_trial, zero_arr])) 
        seq_periods.append([seq_periods[-1][1], seq_periods[-1][1] + s_white_noise.shape[0]])

    # get first 3 airpuff blocks 
    airpuff_trial_dur = 13.5
    airpuff_trial_cs = trials_df[trials_df['trialtype'] == 'TONE_3_LAG_500_AIRPUFF_1']['cs'].to_list()[0:3]
    airpuff_trial_us = trials_df[trials_df['trialtype'] == 'TONE_3_LAG_500_AIRPUFF_1']['us'].to_list()[0:3]

    for trial_idx, trial in enumerate(airpuff_trial_cs):
        spike_counts_airpuff, s_airpuff = get_block(spike_counts, neuromod_activity, trial - 15, trial + airpuff_trial_dur + 15, bin_size)
        us = airpuff_trial_us[trial_idx]
        stim_airpuff = get_airpuff_trial(trial, us, bin_size) 

        s_min = np.minimum(s_min, s_airpuff.min())
        s_max = np.maximum(s_max, s_airpuff.max())
        
        x_train.append(spike_counts_airpuff)
        s_train.append(s_airpuff) 
        stim_arr_train.append(np.hstack([zero_arr, stim_airpuff, zero_arr])) 
        seq_periods.append([seq_periods[-1][1], seq_periods[-1][1] + s_airpuff.shape[0]])

    # get first 4 reward blocks
    for i in range(4):
        # get stimuli 
        stim_arr_rewards, trial_blocks, reward_trial_start = get_reward_block(i, bin_size)
        spike_counts_reward, s_reward = get_block(spike_counts, neuromod_activity,reward_trial_start - 15, reward_trial_start + 215 + 15, bin_size)
        s_min = np.minimum(s_min, s_airpuff.min())
        s_max = np.maximum(s_max, s_airpuff.max())
        
        x_train.append(spike_counts_reward)
        s_train.append(s_reward)
        stim_arr_train.append(np.hstack([zero_arr, stim_arr_rewards, zero_arr]))
        seq_periods.append([seq_periods[-1][1], seq_periods[-1][1] + s_reward.shape[0]])

    x_train = np.hstack(x_train)
    s_train = np.hstack(s_train) 
    print(s_pt_baseline.shape, s_train.shape)
    print(f's_min = {s_min}, s_max = {s_max}')
    stim_arr_train = np.hstack(stim_arr_train) 

    # normalize neuromodulators 
    s_train = (s_train - s_min) / (s_max - s_min)
    if config["shuffle"]:
         print('Randomly shuffling Neuromodulators')
         np.random.shuffle(s_train)
    print(f'Shape of training samples: {x_train.shape}, {s_train.shape}')
    return x_train, s_train, stim_arr_train, seq_periods

if __name__ == '__main__':
    config = {
        "rank": 20,
        "z_score": False, 
        "deconvolve": False, 
        "convolve_spikes": True, 
        "neuromodulation": "additive",
        "dataset": "nk341_mPFC",
        "normalize_neuromod": True, 
        "sim_s": True, 
        "sim_v": False, 
        "gpu": "0",
        "sampling_rate": 30_000, 
        "data_dir": "../../../data/recordings/nk341_mPFC/",
        "out_dir": "../../../models/dtt/",
        "bin_size": 0.05, 
        "bs": 512,
        "fr_threshold": 0.5,
        "epochs": 300,
        "shuffle": False,
        "k": 64
    }

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-n', '--neuromodulation', help='Neuromodulation type')
    parser.add_argument('-g', '--gpu', help='GPU')
    parser.add_argument('-d', '--dataset', help='Dataset type')
    parser.add_argument('-e', '--epochs', help='Number of epochs to train')
    parser.add_argument('-r', '--rank', help='Rank of network')
    parser.add_argument('-s', '--shuffle', help='Shuffle Neuromodulators for control')
    parser.add_argument('-k', '--particles', help='Number of particles')
    parser.add_argument('-t', '--stimuli', help='Add stimuli')
    parser.add_argument('-a', '--activation', help='Activation function (relu, sigmoid, tanh)')

    args = parser.parse_args() 

    if args.neuromodulation is not None: 
        config["neuromodulation"] = args.neuromodulation 
        if 'stim' in config["neuromodulation"]: 
            config["neuromodulation"] = config["neuromodulation"].split('_')[0]
            config["sim_v"] = True 
        if args.stimuli is not None: 
            config["sim_v"] = bool(args.stimuli)
        if config['neuromodulation'] == 'None': config['neuromodulation'] = None
    if config["neuromodulation"] != "additive": 
        config["sim_s"] = False 
    if args.dataset is not None: 
        config["dataset"] = args.dataset 
        
    if config["dataset"] == "nk340_mPFC": 
        config["data_dir"] = "../../../data/recordings/nk340_mPFC/"
    #     config["t_start_post_trial"] = 4700
    #     config["t_end_post_trial"] = 5150
    elif config["dataset"] == "nk339_mPFC": 
        config["data_dir"] = "../../../data/recordings/nk339_mPFC/"
    #     config["t_start_post_trial"] = 4992
    #     config["t_end_post_trial"] = 5622
    elif config["dataset"] == "nk341_BLA":
        config["data_dir"] = "../../../data/recordings/nk341_BLA/"
    #     config["t_start_post_trial"] = 4848
    #     config["t_end_post_trial"] = 5848

    if args.gpu is not None: 
        config["gpu"] = args.gpu 

    if args.epochs:
        config["epochs"] = int(args.epochs)

    if args.rank:
        config["rank"] = int(args.rank)

    if args.particles:
        config["k"] = int(args.particles)

    if args.activation is not None:
        config["activation"] = args.activation

    os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu"]
    if args.shuffle: 
        config["shuffle"] = int(args.shuffle)
    print(config)

    # load data 
    spikes, spike_counts, neuromod_activity, trials_df = load_data(config) 

    # split dataset into train and test samples based on amount of data available 
    data_dur = spike_counts.shape[1] * config['bin_size']
    # first find when the last trial is. Last trials is always odor, which lasts 120 seconds 
    last_trial_timestamp = trials_df['cs'].max() + 120
    baseline_dur = data_dur - last_trial_timestamp
    # split into 75% train and 25% test samples. The tail end of the data will be used for training
    config['t_start_post_trial_train'] = int(data_dur - 0.75 * baseline_dur)
    config['t_end_post_trial_train'] = int(data_dur) 

    print(f'Data duration: {data_dur} seconds')
    print(f'Baseline data duration: {baseline_dur} seconds')
    print(f'Baseline training samples: {baseline_dur * 0.75} seconds')
    print(f'Baseline test samples: {baseline_dur * 0.25} seconds')
    
    # prepare dataset 
    x_train, s_train, stim_arr_train, seq_periods = prepare_dataset(spike_counts, neuromod_activity, trials_df, config)

    # train models 
    seeds = np.arange(0, 10)
    for seed in seeds:
        print(f'Training seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        rank, N = config["rank"], spike_counts.shape[0]
        
        seq_periods_eval = []
        
        task_params = {
            "dur": 100,
            "n_trials": 3000,
            "name": "",
            "dataset_name": config["data_dir"],
        }
        device = torch.device('cuda')
        task = DTTDataset(task_params, 
                            x_train.T, 
                            device=device,
                            s_train=s_train if config["neuromodulation"] is not None else None, 
                            s_test=s_train if config["neuromodulation"] is not None else None,
                            stim_train=stim_arr_train if config["sim_v"] else None,
                            stim_test=stim_arr_train if config["sim_v"] else None,
                            seq_periods=seq_periods,
                            seq_periods_eval=seq_periods_eval,
                            data_eval=None)
        
        enc_params = {"obs_grad": True, "init_scale": 0.1}
        
        # initialise prior
        rnn_params = {
            "clipped": True,
            "train_noise_x": True,  # False
            "train_noise_z": True,
            "train_noise_z_t0": True,
            "init_noise_z": 0.1,
            "init_noise_z_t0": 0.1,
            "init_noise_x": 0.1,
            "scalar_noise_z": "Cov",
            "scalar_noise_x": False,
            "scalar_noise_z_t0": "Cov",
            "identity_readout": True,
            "activation": config["activation"],
            "exp_par": True,
            "shared_tau": 0.9,
            "readout_rates": "currents",
            "train_obs_bias": False,
            "train_obs_weights": False, #True
            "train_latent_bias": False,
            "train_neuron_bias": True,
            "orth": False,
            "m_norm": False,
            "weight_dist": "uniform",
            "weight_scaler": 1,  # /dim_N,
            "initial_state": "trainable",
            "out_nonlinearity": "identity",# "softplus",
            "neuromodulation": config["neuromodulation"], 
            "train_nm_params": True
        }
        # initialise training parameters
        training_params = {
            "smooth_at_eval": 200,
            "run_eval": False,
            "t_forward": 0,
            "lr": 1e-3,
            "step_size": 1, 
            "gamma": 0.998849,
            "lr_end": 1e-4,
            "n_epochs": config["epochs"],
            "grad_norm": 10,
            "eval_epochs": 10,
            "batch_size": config["bs"],
            "cuda": True,
            "smoothing": 20,
            "freq_cut_off": 10000,
            "sim_obs_noise": 0,
            "sim_latent_noise": 1,
            "opt_eps": 1e-8,
            "sim_obs_noise": 0,
            "k": config["k"],
            "sim_v": config["sim_v"],
            "sim_s": config["sim_s"],
            "loss_f": "opt_VGTF",
            "resample": "systematic",  # , multinomial or none"
            "observation_likelihood": "Gauss",  # observation likelihood,
            "neuromodulation": config["neuromodulation"]
        }
        
        dim_x = task.data.shape[1]
        dim_z = rank
        dim_N = N
        dim_u = 9
        dim_s = 1 
        
        VAE_params = {
            "dim_x": dim_x,
            "dim_z": dim_z,
            "dim_u": dim_u if config['sim_v'] else 0,
            "dim_N": dim_N,
            "dim_s": dim_s,
            "enc_architecture": "Inv_Obs", #CNN
            "enc_params": enc_params,
            "prior_architecture": "PLRNN",
            "rnn_params": rnn_params,
            "causal": False
        }
        vae = VAE(VAE_params)

        if config['sim_v']: fname = f"{config['dataset']}_{config['neuromodulation']}_rank_{dim_z}_seed_{seed}_stim_{config['activation']}"
        else: fname = f"{config['dataset']}_{config['neuromodulation']}_rank_{dim_z}_seed_{seed}_{config['activation']}"
            
        train_VAE(vae, 
            training_params, 
            task, 
            sync_wandb=True, 
            out_dir=config["out_dir"], 
            fname=fname)


