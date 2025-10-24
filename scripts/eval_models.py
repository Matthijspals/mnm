
import os 
import sys
import argparse 
sys.path.append('./')

import torch
import numpy as np

from evaluation.kl_Gauss import *
from evaluation.calc_stats import * 

from vi_rnn.utils import *
from vi_rnn.load_data import * 
from vi_rnn.evaluation import * 
from vi_rnn.saving import load_model 


import pickle

import warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

torch.manual_seed(0)
np.random.seed(0)


def evaluate(vae, 
             neuromodulation, 
             x_test, 
             s_test, 
             stim_arr_test, 
             white_noise_trial_avgs,
             airpuff_trial_avgs,
             reward_trial_avgs,
             seq_periods, 
             num_samples=3000, 
             num_trajs=10,
             use_stim=True):
    """
    Evaluates a loaded model on test data samples 
    Args:
        vae: Trained model 
        neuromodulation (string): Neuromodulation type (None, postsynaptic, presynaptic, additive, rank) 
        x_test (torch.tensor; neurons x time): Concatenated tensor of neural activity. Indices belonging to specific
                                                blocks of activity are found in seq_periods array
        s_test (torch.tensor; time): Test neuromodualtion signal 
        stim_arr_test (torch.tensor; stimuli x time): Stimuli train for trials 
        white_noise_trial_avgs (torch.tensor; neurons x time): Trial averaged population activity for white noise trials 
        airpuff_trial_avgs (torch.tensor; neurons x time): Trial averaged population activity for airpuff trials 
        reward_trial_avgs (torch.tensor; neurons x time): Trial averaged population activity for reward trials 
        seq_periods (list(list); indices for different signal blocks. First is baseline, next 4 
                                are white noise trials, then three blocks of airpuffs followed by 3 blocks of reward
        num_trajs (int): number of stochastic trajectories to generate from model 
    """
    # Define results dictionary 
    stat_dict = {}
    # first we compute state-space metrics for baseline period
    x_test_baseline = x_test[:, seq_periods[0][0]: seq_periods[0][1]]
    s_test_baseline = s_test[seq_periods[0][0]: seq_periods[0][1]]
    stim_arr_test_baseline = stim_arr_test[:, seq_periods[0][0]: seq_periods[0][1]] # bunch of zeros 
    if use_stim == False: 
        stim_arr_test_baseline = None 
        stim_arr_test = None 

    kl_divs = []
    was_dists = [] 
    sliced_was_dists = []
    mres = [] 
    for i in range(num_trajs):
        # generate trajectory 
        print('generating trajectory', flush=True)
        traj_gen = generate_trajectory(x_test_baseline, s_test_baseline, stim_arr_test_baseline, vae, neuromodulation)[1]
        print('computing KL-div', flush=True)
        kl_div = compute_KL_divergence(traj_gen.unsqueeze(0), x_test_baseline.unsqueeze(0), n_samples=num_samples)
        print('computing wasserstein', flush=True)
        was_dist = compute_wasserstein(x_test_baseline.T.numpy(), traj_gen.T.numpy(), n_samples=num_samples)
        print('computing sliced wasserstein')
        sliced_was_dist = sliced_wasserstein_distance(x_test_baseline, traj_gen, num_projections=100).mean()
        print('computing mean-rate error', flush=True)
        mre = mean_rate(traj_gen.numpy(), x_test_baseline.numpy())
        print(f'Sample {i}, KL Div: {kl_div}, Wasserstein: {was_dist}, Sliced Wasserstein: {sliced_was_dist}, mean-rate: {mre}', flush=True)

        kl_divs.append(kl_div) 
        was_dists.append(was_dist) 
        sliced_was_dists.append(sliced_was_dist)
        mres.append(mre) 
    
    stat_dict['kl divs'] = kl_divs
    stat_dict['wasserstein'] = was_dists
    stat_dict['sliced wasserstein'] = sliced_was_dist
    stat_dict['mres'] = mres

    population_trial_avgs = [white_noise_trial_avgs, airpuff_trial_avgs, reward_trial_avgs]
    # Now we compute the R between the mean stimulus response and the mean actual activity over trials 
    for i, trial in enumerate(['white noise', 'airpuff', 'reward']):
        r = compute_R(vae, 
                    neuromodulation, 
                    x_test, 
                    s_test, 
                    stim_arr_test, 
                    population_trial_avgs[i],
                    trial, 
                    seq_periods)
        stat_dict[f'R {trial}'] = r 
        print(f'R {trial} = {r}')

    return stat_dict 


if __name__ == '__main__':
    config = {
        "rank": 3,
        "z_score_neuromod": False,
        "z_score_neurons": False, 
        "min_max_norm_neuromod": True,
        "deconvolve": False, 
        "convolve_spikes": True, 
        "neuromodulation": "additive",
	    "activation": "clipped_relu",
        "dataset": "nk341_mPFC",
        "normalize_neuromod": True,
        'center_neuromod': False,
        'load_physiology': False, 
        "sim_s": True, 
        "sim_v": False, 
        "sampling_rate": 30_000, 
        "data_dir": "data/recordings/nk341_mPFC/",
        "out_dir": "results/model_evals/",
        "bin_size": 0.05, 
        'fr_threshold': 0.5,
        'threshold_neurons': True,
        "bs": 512,
        "epochs": 400,
        "shuffle": False,
        "k": 64,
        "dales_law": False 
    }

    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('-n', '--neuromodulation', help='Neuromodulation type')
    parser.add_argument('-d', '--dataset', help='Dataset type')
    parser.add_argument('-r', '--rank', help='Rank of network')
    parser.add_argument('-s', '--shuffle', help='Shuffle Neuromodulators for control')
    parser.add_argument('-k', '--particles', help='Number of particles')
    parser.add_argument('-t', '--stim', help='Add stimuli')
    parser.add_argument('-a', '--activation', help='Activation function')
    parser.add_argument('-l', '--dales_law', help='Apply Dale\'s law')
    parser.add_argument('-z', '--bin_size', help='Bin size')

    args = parser.parse_args() 

    config["neuromodulation"] = args.neuromodulation 
    if 'stim' in config["neuromodulation"]: 
        config["neuromodulation"] = config["neuromodulation"].split('_')[0]
        config["sim_v"] = True 
    if args.stim is not None and args.stim == 'True': 
        config["sim_v"] = True
    if config['neuromodulation'] == 'None': config['neuromodulation'] = None
    if config["neuromodulation"] != "additive": 
        config["sim_s"] = False 
    
    config["dataset"] = args.dataset 
    if config["dataset"] == "nk340_mPFC": 
        config["data_dir"] = "data/recordings/nk340_mPFC/"
    elif config["dataset"] == "nk339_mPFC": 
        config["data_dir"] = "data/recordings/nk339_mPFC/"

    if args.dales_law is not None:
        config["dales_law"] = bool(args.dales_law)

    if args.bin_size is not None: 
        config["bin_size"] = float(args.bin_size) 
        
    config["rank"] = int(args.rank)
    if args.activation:
        config["activation"] = args.activation

    print(config, flush=True)

    spikes, spike_counts, neuromod_activity, trials_df, cell_types = load_data(config)
    # split dataset into train and test samples based on amount of data available 
    data_dur = spike_counts.shape[1] * config['bin_size']
    # first find when the last trial is. Last trials is always odor, which lasts 120 seconds 
    last_trial_timestamp = trials_df['cs'].max() + 120
    baseline_dur = data_dur - last_trial_timestamp
    # split into 75% train and 25% test samples. The tail end of the data will be used for training
    config['baseline_test_start'] = int(data_dur - baseline_dur)
    config['baseline_test_end'] = int(data_dur - 0.75 * baseline_dur) 

    config['baseline_train_start'] = config['baseline_test_end']
    config['baseline_train_end'] = int(data_dur)

    print(f'Data duration: {data_dur}, Baseline duration: {baseline_dur}')
    print(f"Test start timestamp: {config['baseline_test_start']}, end timestamp: {config['baseline_test_end']}", flush=True)
    print(f"Train start timestamp: {config['baseline_train_start']}, end timestamp: {config['baseline_train_end']}", flush=True)

    if args.shuffle: 
        config["shuffle"] = int(args.shuffle)

    # prepare dataset 
    x_test, s_test, stim_arr_test, seq_periods, white_noise_trial_avgs, airpuff_trial_avgs, reward_trial_avgs = prepare_dataset(spike_counts, 
                                                                              neuromod_activity,
                                                                              trials_df, 
                                                                              config,
                                                                              test=True)
    # load the training samles for normalizing s_test 
    _, s_train, _, _, _, _, _ = prepare_dataset(spike_counts, 
                                                neuromod_activity,
                                                trials_df, 
                                                config,
                                                test=False)
    print(f's_train max: {s_train.max()}, s_train min: {s_train.min()}')
    # min-max normalize the neuromodulation signal 
    if config["min_max_norm_neuromod"]:
        s_test = (s_test - s_train.min()) / (s_train.max() - s_train.min())
    
    # convert to torch tensors 
    x_test = torch.from_numpy(x_test).to(torch.float32) 
    s_test = torch.from_numpy(s_test).to(torch.float32)
    stim_arr_test = torch.from_numpy(stim_arr_test).to(torch.float32) 
    white_noise_trial_avgs = torch.from_numpy(white_noise_trial_avgs).to(torch.float32)
    airpuff_trial_avgs = torch.from_numpy(airpuff_trial_avgs).to(torch.float32)
    reward_trial_avgs = torch.from_numpy(reward_trial_avgs).to(torch.float32)
 
    kl_divs = [] # kl-divergence scores
    mres = [] # mean-rate errors 
    was_dists = [] # wasserstein distances 
    r_vals = [] # R values for stimulus trials 
   
    for seed in range(0, 10):
        vae, params, task_params, training_params = load_model(f'models/all/{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}')
        
        stat_dict = evaluate(vae, 
                            config["neuromodulation"], 
                            x_test, 
                            s_test, 
                            stim_arr_test,
                            white_noise_trial_avgs,
                            airpuff_trial_avgs,
                            reward_trial_avgs, 
                            seq_periods,
                            use_stim=config['sim_v'])
        
        # save dictionary as pickle 
        result_dir = f'{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}/stat_dict.pkl'
        file_path = os.path.join(config["out_dir"], result_dir)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
            