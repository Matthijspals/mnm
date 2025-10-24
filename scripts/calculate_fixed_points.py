import sys
sys.path.append("./")

import os 
import argparse

import numpy as np 
import pickle

from vi_rnn.saving import load_model
from vi_rnn.utils import *
from vi_rnn.load_data import * 


if __name__ == '__main__':
    config = {
        'neuromodulation': 'postsynaptic',
        'rank': 3,
        'dataset': 'nk341_mPFC',
        'seed': 0,
        'start': 0,
        'end': 1
    }

    parser = argparse.ArgumentParser(description='calculate') 
    parser.add_argument('-n', '--neuromodulation', help='Neuromodulation Type') 
    parser.add_argument('-r','--rank', help='Rank of the network')
    parser.add_argument('-d', '--dataset', help='Dataset type') 
    parser.add_argument('-b', '--seed', help='Seed') 
    parser.add_argument('-s', '--start', help='Start') 
    parser.add_argument('-e', '--end', help='End') 

    args = parser.parse_args() 
    if args.neuromodulation is not None: 
        config['neuromodulation'] = args.neuromodulation 
    if args.rank is not None:     
        config['rank'] = int(args.rank)
    if args.dataset is not None: 
        config['dataset'] = args.dataset 
    if args.seed is not None: 
        config['seed'] = int(args.seed)
    if args.start is not None: 
        config['start'] = int(args.start)
    if args.end is not None: 
        config['end'] = int(args.end)

    print(config)
    print(f'Loading Models', flush=True)
    model, params, task_params, training_params = load_model(f'./models/all/{config["dataset"]}_{config["neuromodulation"]}_rank_{config["rank"]}_activation_clipped_relu_seed_{config["seed"]}_stim_True_binsize_0_05_daleslaw_False')
    # Extract parameters 
    a, V, U, B, I = get_loadings(model)

    A = model.rnn.transition.A.detach().numpy()
    hz = model.rnn.transition.hz.detach().numpy()
    W_inp = model.rnn.transition.Wu.detach().numpy()
    h = model.rnn.transition.h.detach().numpy()
    R, N = V.shape

    # Orthogonalize
    # U, S, V = np.linalg.svd(U @ V)
    # U = U[:, :R]
    # V = (V[:R].T*S[:R]).T

    # use orthogonalized low-rank matrices 
    # model.rnn.transition.m = torch.nn.Parameter(torch.from_numpy(U))
    # model.rnn.transition.n = torch.nn.Parameter(torch.from_numpy(V))

    V_new, U_new, h_new = convert_clipped(V.T,U,h)
    # test_conversion(V, V_new, U, U_new, h, h_new, N, R, s)
    print(f'Doing fixed point sweep', flush=True)
    s_vals, fixed_points = fixed_point_sweep(V_new, U_new, hz, h_new, a, A, W_inp, neuromodulation=config['neuromodulation'], start=config["start"], end=config["end"], num_points=100)
    print('Done')
    with open(f'./results/pickle/{config["neuromodulation"]}_rank_{config["rank"]}_s_vals_{config["dataset"]}_seed_{config["seed"]}_bifurcation_2.pkl', 'wb') as f: 
        pickle.dump(s_vals, f)
    with open(f'./results/pickle/{config["neuromodulation"]}_rank_{config["rank"]}_fixed_points_{config["dataset"]}_seed_{config["seed"]}_bifurcation_2.pkl', 'wb') as f: 
        pickle.dump(fixed_points, f) 
