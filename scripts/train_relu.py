import os 
import sys
import argparse 

sys.path.append("./")
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.load_data import *
from vi_rnn.utils import *
from vi_rnn.evaluation import * 
from vi_rnn.datasets import DTTDataset
CUDA = True

if __name__ == '__main__':
    config = {
        "rank": 6,
        "z_score_neuromod": False, 
        "min_max_norm_neuromod": True,
        "z_score_neurons": False,
        "deconvolve": False, 
        "convolve_spikes": True, 
        "neuromodulation": "postsynaptic",
        "dataset": "nk339_mPFC",
        "normalize_neuromod": True, 
        'center_neuromod': False,
        "sim_s": False, 
        "sim_v": True, 
        "gpu": "0",
        "sampling_rate": 30_000, 
        "data_dir": "data/nk339/",
        "out_dir": "models/all/",
        "bin_size": 0.05, 
        "bs": 32, 
        "threshold_neurons": True,
        "fr_threshold": 0.5,
        "epochs": 1500,
        "shuffle": False,
        "k": 64,
        "dales_law": False,
        "activation": "relu",
        "learning_rate": 1e-3,
        "load_physiology": False,
        "seed": None,
        "shared_tau": 0.9,
        "train_alpha": True,
        "stim":True, #what is the right setting for this
        "ed_ratio": 0.5,
        'center_data': False
    }



    print(config, flush=True)

    # load data 
    spikes, spike_counts, neuromod_activity, trials_df, cell_types = load_data(config) 
    if cell_types is not None:
        cell_types = torch.from_numpy(cell_types).to(torch.float32)
    # split dataset into train and test samples based on amount of data available 
    data_dur = spike_counts.shape[1] * config['bin_size']
    # first find when the last trial is. Last trials is always odor, which lasts 120 seconds 
    last_trial_timestamp = trials_df['cs'].max() + 120
    baseline_dur = data_dur - last_trial_timestamp
    # split into 75% train and 25% test samples. The tail end of the data will be used for training
    config['baseline_train_start'] = int(data_dur - 0.75 * baseline_dur)
    config['baseline_train_end'] = int(data_dur) 

    print(f'Data duration: {data_dur} seconds')
    print(f'Baseline data duration: {baseline_dur} seconds')
    print(f'Baseline training samples: {baseline_dur * 0.75} seconds')
    print(f'Baseline test samples: {baseline_dur * 0.25} seconds')
    
    # prepare dataset 
    x_train, s_train, stim_arr_train, seq_periods, _, _, _ = prepare_dataset(spike_counts, neuromod_activity, trials_df, config, test=False, time_delta=15)
    
    # normalize neuromodulators 
    if config["z_score_neuromod"] == False and config["min_max_norm_neuromod"]:
        s_train = (s_train - s_train.min()) / (s_train.max() - s_train.min()) 

    # train models 
    if config["seed"] is not None:
        seeds = [config["seed"]]
    else:
        seeds = np.random.randint(0, 10000, size=1).tolist()
        print("SETTING SEEDS TO: ", seeds)
    for seed in seeds:
        print(f'Training seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        rank, N = config["rank"], spike_counts.shape[0]
        
        seq_periods_eval = []
        
        task_params = {
            "dur": 100,
            "n_trials": 5000,
            "name": "",
            "dataset_name": config["data_dir"],
        }
        if CUDA:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
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
            "clipped": False,
            "train_noise_x": True,  # False
            "train_noise_z": True,
            "train_noise_z_t0": True,
            "init_noise_z": 0.1,
            "init_noise_z_t0": 0.1,
            "init_noise_x": 0.1,
            "scalar_noise_z":False,#"Cov",# "Cov",
            "scalar_noise_x": False,
            "scalar_noise_z_t0": False,#"Cov",#"Cov",
            "identity_readout": True,
            "activation": config["activation"],
            "exp_par": True,
            "shared_tau": config["shared_tau"],
            "readout_rates": "rates",
            "train_obs_bias": False,
            "train_obs_weights": True, 
            "train_latent_bias": False,
            "train_neuron_bias": True, # TODO: return to True
            "orth": False,
            "m_norm": False,
            "weight_dist": "uniform",
            "weight_scaler": .4,  # /dim_N,
            "initial_state": "trainable",
            "out_nonlinearity": "identity",# "softplus",
            "neuromodulation": config["neuromodulation"], 
            "train_nm_params": True,
            "train_alpha": config["train_alpha"]
        }
        # initialise training parameters
        training_params = {
            "smooth_at_eval": 200,
            "run_eval": False,
            "t_forward": 0,
            "lr": config["learning_rate"],
            "step_size": 1, 
            "gamma": 0.998849,
            "lr_end": 1e-4,
            "n_epochs": config["epochs"],
            "grad_norm": 10,
            "eval_epochs": 10,
            "batch_size": config["bs"],
            "cuda": CUDA,
            "smoothing": 20,
            "freq_cut_off": 10000,
            "sim_obs_noise": 0,
            "sim_latent_noise": 1,
            "opt_eps": 1e-8,
            "sim_obs_noise": 0,
            "k": config["k"],
            "sim_v": config["sim_v"],
            "sim_s": config["sim_s"],
            "loss_f": "VGTF",
            "resample": "systematic",  # , multinomial or none"
            "observation_likelihood": "Gauss",  # observation likelihood,
            "neuromodulation": config["neuromodulation"],
            "ed_ratio": config["ed_ratio"]
        }
        
        dim_x = task.data.shape[1]
        dim_z = rank
        dim_N = N
        dim_u = 9
        dim_s = 1 
        

        enc_params ={
            "init_kernel_sizes": [4, 2, 2],
            "nonlinearity": "gelu",
            "n_channels": [32, 16],
            "init_scale": 0.05,
            "padding_location": "acausal",
            "constant_var": False,
            "padding_mode": "constant"  # reflect #reflect # constant reflect replicate or circular
        }

        VAE_params = {
            "dim_x": dim_x,
            "dim_z": dim_z,
            "dim_u": dim_u if config['sim_v'] else 0,
            "dim_N": dim_N,
            "dim_s": dim_s,
            "enc_architecture": "CNN",#Inv_Obs",#CNN", #CNN
            "enc_params": enc_params,
            "prior_architecture": "PLRNN",
            "rnn_params": rnn_params,
            "causal": False,
            "cell_types": cell_types
        }
        vae = VAE(VAE_params)

        fname = f'{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}'

        train_VAE(vae, 
            training_params, 
            task, 
            sync_wandb=True, 
            out_dir=config["out_dir"], 
            fname=fname)


