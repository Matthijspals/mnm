from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
from pathlib import Path

class MockDataset(Dataset):
    def __init__(self, task_params, data, task_input, s, data_eval=None):
        self.task_params = task_params
        self.data = data 
        self.data_eval = self.data
        self.s = s 

        self.task_input = task_input
        
        self.dur = task_params['dur']
        self.n_trials = task_params['n_trials']

    def __len__(self):
        return self.n_trials 
    
    def __getitem__(self, idx):
        """
        Return a trial of length self.dur 
        Args: 
            idx (int): trial 
        Returns: 
            trial (torch.tensor; dim_x x self.dur): trial of length self.dur 
            input (torch.tensor; n_inp x self.dur): optional input on which the model is conditioned  
            s (torch.tensor; dim_s x self.dur): Neuromodulation signal 
        """
        if self.s is None: 
            return self.data[idx].T, \
                self.task_input[idx].T

        return self.data[idx].T, \
                self.task_input[idx].T, \
                self.s[idx].T