import os
import numpy as np
from synapse_model import memsynapse
from memory_net import memory

R_thres = 11e6
R_bounds = [1e3, 1e9]
synapse = memsynapse(R_thres, R_bounds)

"""
Images experiment; follows same principles with random_stream but uses fixed memory signals
"""
no_synapses = 10000
signals_path = "../data/network/binary_images/signal_source/signals_stm_2.csv"
export_dir = "../data/network/binary_images/generated_data"
net = memory(no_synapses, synapse, R_thres, param_paths=None)
t = 10
signal_intensity = [3,1,1]
no_runs = 3
no_events = np.sum(signal_intensity)
trace_index = [0,3,4]
signal_noise = [0, 0.02, 0.05, 0.1, 0.2]
experiment_dir, noise_dirs = net.application_experiment(signals_path, t, no_runs, signal_intensity, ltm_noise=signal_noise,  export_dir=export_dir)
for folder in noise_dirs:
    dir = os.path.join(experiment_dir, folder)
    net.correlation_overlaps(dir, t, no_synapses, no_events, no_runs, trace_index)
