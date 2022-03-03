import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import pprint
from synapse_model import memsynapse
from memory_net import memory

R_thres = 11e6
R_bounds = [1e3, 1e9]
synapse = memsynapse(R_thres, R_bounds)

"""
Show memory state snapshots
"""
parent_path = "../data/network/binary_images/generated_data"
experiment_dir = "030322_131244"
data_folder = 'ltm_noise_0.1'
data_history = "history_N_10000_run_0_e_5_t_10.csv"
data_path = os.path.join(parent_path, experiment_dir, data_folder, data_history)
test = memory(1, synapse, R_thres)
time_stamps = np.array([20, 30, 30.1, 30.2, 30.3, 30.5, 31, 33, 35, 37, 39, 40, 40.1, 40.2, 40.3, 40.5, 41, 43, 45, 47])
test.plot_image_snaps(data_path, time_stamps)

"""
Compute LTM signal overlap
"""
parent_path = "../data/network/binary_images/generated_data"
experiment_dir = # Add experiment directory (time_tag)
experiment_path = os.path.join(parent_path, experiment_dir)
folders = ['ltm_noise_0', 'ltm_noise_0.02', 'ltm_noise_0.05', 'ltm_noise_0.1', 'ltm_noise_0.2']
test = memory(1, synapse, R_thres)
t = 10
no_synapses = 10000
no_events = 5
ltm_noise = [0, 0.02, 0.05, 0.1, 0.2]
no_runs = 10
signal_id = 0
test.plot_image_ltm_overlap(experiment_path, folders, t, no_synapses, no_events, signal_id, ltm_noise)
