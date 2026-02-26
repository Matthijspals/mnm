
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
from vi_rnn.generate import * 

from scipy.signal import convolve
from scipy.signal.windows import gaussian 

import pickle

import warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

torch.manual_seed(0)
np.random.seed(0)


def smooth_spikes(spikes, T, kernel_size, sigma): 
    gaussian_kernel = gaussian(kernel_size, sigma)
    # Keep only the causal part (including the current time)
    trajs = []
    gaussian_kernel /= gaussian_kernel.sum()
    for n in range(spikes.shape[0]):
        original_length = T
        padded = torch.nn.functional.pad(spikes[n],
                            (kernel_size - 1, 0),
                            mode='constant',
                            value=0)
        # Convolve and take only the 'valid' part (no future context)
        smoothed = convolve(padded.numpy(), gaussian_kernel, mode='valid')[:original_length]
        trajs.append(smoothed)
    trajs = torch.tensor(trajs).to(torch.float32)
    return trajs

def center_trajs(trajs, mean_rates): 
    trajs = trajs - mean_rates
    return trajs 


def evaluate(vae, 
             neuromodulation, 
             x_test, 
             s_test, 
             stim_arr_test, 
             white_noise_trial_avgs,
             airpuff_trial_avgs,
             reward_trial_avgs,
             seq_periods, 
             mean_rates,
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
    kernel_size = 25 
    sigma = 5 

    # Define results dictionary 
    stat_dict = {}
    # first we compute state-space metrics for baseline period
    x_test_baseline = x_test[:, seq_periods[0][0]: seq_periods[0][1]]
    s_test_baseline = s_test[seq_periods[0][0]: seq_periods[0][1]].view(1, 1, -1)
    stim_arr_test_baseline = stim_arr_test[:, seq_periods[0][0]: seq_periods[0][1]].unsqueeze(0) # bunch of zeros 

    N, T = x_test_baseline.shape

    if use_stim == False: 
        stim_arr_test_baseline = None 
        stim_arr_test = None 

    kl_divs, kl_divs_rates = [], []
    was_dists, was_dists_rates = [], []
    sliced_was_dists, sliced_was_dists_rates = [], []
    mres, mres_rates = [], [] 

    for i in range(num_trajs):
        # generate trajectory 
        print('generating trajectory', flush=True)

        if config["obs"] == "poisson": 
            # Some trajectories are bad (NaNs). Generate until you get a valid trajectory 
            traj_ok = False
            while(traj_ok == False):
                # try:
                #_, _, lmd = generate(vae, x_test_baseline, s_test_baseline, stim_arr_test_baseline, T, sim_s=config["sim_s"])
                #lmd = lmd.reshape(N, -1) 
                #traj_gen = torch.poisson(lmd) 
                _, _, traj_gen,spike = generate(vae, x_test_baseline, s_test_baseline, stim_arr_test_baseline, x_test_baseline.shape[1])
                traj_ok = True
                traj_gen = traj_gen.reshape(N, -1)
                #except:
                #    print('Generating a new trajectory')            
                
        elif config["obs"] == "nonlinear_gauss" or config["obs"] == "gauss":
            _, _, traj_gen = generate(vae, x_test_baseline, s_test_baseline, stim_arr_test_baseline, T, sim_s=config["sim_s"])
            print(traj_gen.shape)
            traj_gen = traj_gen.reshape(N, -1) 

        if config["obs"] == 'nonlinear_gauss' or config["obs"] == "poisson":
            # smooth if spikes
            if config["obs"] == "poisson":
                traj_gen = smooth_spikes(traj_gen, T, kernel_size, sigma)
                x_test_baseline_smoothed = smooth_spikes(x_test_baseline, T, kernel_size, sigma)
            else: x_test_baseline_smoothed = x_test_baseline 

            # compute the stats in rate space 
            mre_rate_space = mean_rate(traj_gen.numpy(), x_test_baseline_smoothed.numpy()) 
            print('computing KL-div in rate space', flush=True)
            kl_div_rate_space = compute_KL_divergence(traj_gen.unsqueeze(0), x_test_baseline_smoothed.unsqueeze(0), n_samples=num_samples)
            print('computing wasserstein', flush=True)
            was_dist_rate_space = compute_wasserstein(x_test_baseline_smoothed.T.numpy(), traj_gen.T.numpy(), n_samples=num_samples)
            print('computing sliced wasserstein')
            sliced_was_dist_rate_space = sliced_wasserstein_distance(x_test_baseline_smoothed, traj_gen, num_projections=100).mean()

            # center by global mean and compute stats in mean-centered space
            x_test_baseline_smoothed = center_trajs(x_test_baseline_smoothed, mean_rates)
            traj_gen = center_trajs(traj_gen, mean_rates) 
            
            print('computing mean-rate error', flush=True)
            mre = mean_rate(traj_gen.numpy(), x_test_baseline_smoothed.numpy())
            print('computing KL-div', flush=True)
            kl_div = compute_KL_divergence(traj_gen.unsqueeze(0), x_test_baseline_smoothed.unsqueeze(0), n_samples=num_samples)
            print('computing wasserstein', flush=True)
            was_dist = compute_wasserstein(x_test_baseline_smoothed.T.numpy(), traj_gen.T.numpy(), n_samples=num_samples)
            print('computing sliced wasserstein')
            sliced_was_dist = sliced_wasserstein_distance(x_test_baseline_smoothed, traj_gen, num_projections=100).mean()

        elif config["obs"] == "gauss": 
            print('computing KL-div', flush=True)
            kl_div = compute_KL_divergence(traj_gen.unsqueeze(0), x_test_baseline.unsqueeze(0), n_samples=num_samples)
            print('computing wasserstein', flush=True)
            was_dist = compute_wasserstein(x_test_baseline.T.numpy(), traj_gen.T.numpy(), n_samples=num_samples)
            print('computing sliced wasserstein')
            sliced_was_dist = sliced_wasserstein_distance(x_test_baseline, traj_gen, num_projections=100).mean()
            print('computing mean-rate error', flush=True)
            mre = mean_rate(traj_gen.numpy(), x_test_baseline.numpy())

            # compute stats in rate space 
            x_test_baseline_rates = x_test_baseline + mean_rates 
            traj_gen_rates = traj_gen + mean_rates 
            mre_rate_space = mean_rate(traj_gen_rates.numpy(), x_test_baseline_rates.numpy()) 
            print('computing KL-div in rate space', flush=True)
            kl_div_rate_space = compute_KL_divergence(traj_gen_rates.unsqueeze(0), x_test_baseline_rates.unsqueeze(0), n_samples=num_samples)
            print('computing wasserstein', flush=True)
            was_dist_rate_space = compute_wasserstein(x_test_baseline_rates.T.numpy(), traj_gen_rates.T.numpy(), n_samples=num_samples)
            print('computing sliced wasserstein')
            sliced_was_dist_rate_space = sliced_wasserstein_distance(x_test_baseline_rates, traj_gen_rates, num_projections=100).mean()

        print(f'Sample {i}, KL Div: {kl_div}, KL Div rates: {kl_div_rate_space} \
                Wasserstein: {was_dist}, Wasserstein rate space: {was_dist_rate_space} \
                Sliced Wasserstein: {sliced_was_dist}, Sliced Wasserstein rate space: {sliced_was_dist_rate_space}, \
                mean-rate: {mre}, mean-rate rate-space: {mre_rate_space}', flush=True
                )
        
        kl_divs.append(kl_div) 
        kl_divs_rates.append(kl_div_rate_space)
        was_dists.append(was_dist) 
        was_dists_rates.append(was_dist_rate_space)
        sliced_was_dists.append(sliced_was_dist)
        sliced_was_dists_rates.append(sliced_was_dist_rate_space)
        mres.append(mre) 
        mres_rates.append(mre_rate_space)
    
    stat_dict['kl divs'] = kl_divs
    stat_dict['kl_divs_rates'] = kl_divs_rates
    stat_dict['wasserstein'] = was_dists
    stat_dict['wasserstein_rates'] = was_dists_rates
    stat_dict['sliced wasserstein'] = sliced_was_dists
    stat_dict['sliced_wasserstein_rates'] = sliced_was_dists_rates
    stat_dict['mres'] = mres
    stat_dict['mres_rates'] = mres_rates 

    # print mean of all stats:
    print(f'Mean KL Div: {np.mean(kl_divs)}, \
            Mean KL Div rates: {np.mean(kl_divs_rates)}, \
            Mean Wasserstein: {np.mean(was_dists)}, \
            Mean Wasserstein rates: {np.mean(was_dists_rates)}, \
            Mean Sliced Wasserstein: {np.mean(sliced_was_dists)}, \
            Mean Sliced Wasserstein rates: {np.mean(sliced_was_dists_rates)} \
            Mean MRE: {np.mean(mres)} \
            Mean MRE rates: {np.mean(mres_rates)}',
            flush=True)

    population_trial_avgs = [white_noise_trial_avgs, airpuff_trial_avgs, reward_trial_avgs]
    # Now we compute the R between the mean stimulus response and the mean actual activity over trials 
    for i, trial in enumerate(['white noise', 'airpuff', 'reward']):
        r_mean, r_std, r2_mean, r2_std = compute_R(vae, 
                    neuromodulation, 
                    x_test, 
                    s_test, 
                    stim_arr_test, 
                    population_trial_avgs[i],
                    trial, 
                    seq_periods,
                    sim_s=config["sim_s"],
                    obs=config["obs"])
        stat_dict[f'R {trial} mean'] = r_mean.item() 
        stat_dict[f'R {trial} std'] = r_std.item() 

        # print(f'R {trial} = {r}')

    return stat_dict 


if __name__ == '__main__':
    config = {
        "rank": 2,
        "z_score_neuromod": False,
        "z_score_neurons": False, 
        "min_max_norm_neuromod": True,
        "deconvolve": False, 
        "convolve_spikes": True, 
        "zero_pad": True,
        "neuromodulation": "postsynaptic",
	    "activation": "relu",
        "dataset": "nk339_mPFC",
        "normalize_neuromod": True,
        'center_neuromod': False,
        'load_physiology': False, 
        "sim_s": False, 
        "sim_v": True, 
        "sampling_rate": 30_000, 
        "data_dir": "data/recordings/nk339_mPFC/",
        "out_dir": "results/model_evals/",
        "bin_size": 0.05, 
        'fr_threshold': 0.5,
        'threshold_neurons': True,
        "bs": 512,
        "epochs": 400,
        "shuffle": False,
        "k": 64,
        "dales_law": False,
        'center_data': False,
        "obs": "poisson",
        "shift": -4, 
    }

    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--neuromodulation', help='Neuromodulation type', default='postsynaptic')
    parser.add_argument('--dataset', help='Dataset type',default='nk339_mPFC')
    parser.add_argument('--rank', help='Rank of network',default=2)
    parser.add_argument('--shuffle', help='Shuffle Neuromodulators for control', default=False)
    parser.add_argument('--particles', help='Number of particles', default=64)
    parser.add_argument('--no_stim', help='Remove stimuli', default=False, action='store_true')
    parser.add_argument('--activation', help='Activation function', default='relu')
    parser.add_argument('--dales_law', help='Apply Dale\'s law', default=False, action='store_true')
    parser.add_argument('--bin_size', help='Bin size', default=0.05)
    parser.add_argument('--seed', help='Random seed', default=0)
    parser.add_argument('--obs', help='Observation function', default="poisson")
    parser.add_argument('--shift', help='Neuromodulator shift', default=-4) 
    parser.add_argument('--z_score', help='Z-score neuromod', action='store_true')

    args = parser.parse_args() 
    #print(args.seed)
    config["neuromodulation"] = args.neuromodulation 
    if args.no_stim: 
        config["sim_v"] = False
    if config['neuromodulation'] == 'None': config['neuromodulation'] = None
    
    config["dataset"] = args.dataset 
    if config["dataset"] == "nk340_mPFC": 
        config["data_dir"] = "data/recordings/nk340_mPFC/"
    elif config["dataset"] == "nk339_mPFC": 
        config["data_dir"] = "data/recordings/nk339_mPFC/"
    elif config["dataset"] == "nk341_mPFC": 
        config["data_dir"] = "data/recordings/nk341_mPFC/"
    elif config["dataset"] == "nk341_BLA":
        config["data_dir"] = "data/recordings/nk341_BLA/"
    elif config["dataset"] == "nk340_BLA":
        config["data_dir"] = "data/recordings/nk340_BLA/"
    elif config["dataset"] == "nk341_thalamus":
        config["data_dir"] = "data/recordings/nk341_thalamus/"

    if args.dales_law:
        config["dales_law"] = True

    if args.bin_size is not None: 
        config["bin_size"] = float(args.bin_size) 
    
    if args.obs is not None: 
        config["obs"] = args.obs
        if config["obs"] == "nonlinear_gauss" or config["obs"] == "poisson": 
            config["center_data"] = False 
        if config["obs"] == "poisson":
            config["convolve_spikes"] = False 
    
    if args.shift is not None: 
        config["shift"] = int(args.shift)

    if args.z_score: 
        config['z_score_neuromod'] = True 
        config['min_max_norm_neuromod'] = False 
        
    config["rank"] = int(args.rank)
    if args.activation:
        config["activation"] = args.activation

    print(config, flush=True)

    spikes, spike_counts, neuromod_activity, trials_df, cell_types = load_data(config)
    print(f'Shifting neuromodulator signal by {config["shift"]}')
    neuromod_activity = np.roll(neuromod_activity, config["shift"])
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
    x_test, s_test, stim_arr_test, seq_periods_test, white_noise_trial_avgs, airpuff_trial_avgs, reward_trial_avgs = prepare_dataset(spike_counts, 
                                                                              neuromod_activity,
                                                                              trials_df, 
                                                                              config,
                                                                              test=True)
    # load the training samles for normalizing s_test 
    x_train, s_train, _, seq_periods, _, _, _ = prepare_dataset(spike_counts, 
                                                neuromod_activity,
                                                trials_df, 
                                                config,
                                                test=False)
    # min-max normalize the neuromodulation signal 
    if config["min_max_norm_neuromod"]:
        s_test = (s_test - s_train.min()) / (s_train.max() - s_train.min())
    elif config["z_score_neuromod"]:
        s_test = (s_test - s_train.mean()) / s_train.std() 
        s_train = (s_train - s_train.mean()) / s_train.std()
    print(f's_train max: {s_train.max()}, s_train min: {s_train.min()}')

    x_train_baseline = x_train[:, seq_periods[0][0]:seq_periods[0][1]]
    mean_rates = x_train_baseline.mean(axis=1, keepdims=True)
    print("X BASELINE SHAPE: ", x_train_baseline.shape)
    if config["center_data"]:
        x_test = x_test - mean_rates
        white_noise_trial_avgs = white_noise_trial_avgs - mean_rates 
        airpuff_trial_avgs = airpuff_trial_avgs - mean_rates
        reward_trial_avgs = reward_trial_avgs - mean_rates 
        
    # convert to torch tensors 
    x_test = torch.from_numpy(x_test).to(torch.float32) 
    mean_rates = torch.from_numpy(mean_rates).to(torch.float32)
    s_test = torch.from_numpy(s_test).to(torch.float32)
    stim_arr_test = torch.from_numpy(stim_arr_test).to(torch.float32) 
    white_noise_trial_avgs = torch.from_numpy(white_noise_trial_avgs).to(torch.float32)
    airpuff_trial_avgs = torch.from_numpy(airpuff_trial_avgs).to(torch.float32)
    reward_trial_avgs = torch.from_numpy(reward_trial_avgs).to(torch.float32)
   
    for seed in [9291]:#range(0, 1):
        #models/all/nk339_mPFC_postsynaptic_rank_8_activation_relu_seed_6007_stim_True_binsize_0_05_daleslaw_False_obs_poisson_shift_-4
        #models/all/
        #nk339_mPFC_postsynaptic_rank_2_activation_relu_seed_9291_stim_True_binsize_0_05_daleslaw_False_obs_poisson_shift_-4
        #nk339_mPFC_postsynaptic_rank_1_activation_relu_seed_4659_stim_True_binsize_0_05_daleslaw_False_obs_poisson_shift_-4
        print(f'models/all/{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}_obs_{config["obs"]}_shift_{config["shift"]}')
        vae, params, task_params, training_params = load_model(f'models/all/{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}_obs_{config["obs"]}_shift_{config["shift"]}')
      
        #vae, params, task_params, training_params = load_model(f'models/all/{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}_obs_{config["obs"]}_shift_{config["shift"]}')
        
        stat_dict = evaluate(vae, 
                            config["neuromodulation"], 
                            x_test, 
                            s_test, 
                            stim_arr_test,
                            white_noise_trial_avgs,
                            airpuff_trial_avgs,
                            reward_trial_avgs, 
                            seq_periods_test,
                            mean_rates,
                            use_stim=config['sim_v'])
        
        # save dictionary as pickle 
        result_dir = f'{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_{config["activation"]}_seed_{seed}_stim_{config["sim_v"]}_binsize_{str(config["bin_size"]).replace(".", "_")}_daleslaw_{config["dales_law"]}_obs_{config["obs"]}_shift_{config["shift"]}/stat_dict.pkl'
        file_path = os.path.join(config["out_dir"], result_dir)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f'saving results to: {result_dir}')
        with open(file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        print('results saved')
            