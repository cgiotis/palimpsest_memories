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
Palimpsest STM lifetime
"""
parent_path = "../data/network/random_memories/generated_data"
experiment_path = "030322_181905"
data_path = os.path.join(parent_path, experiment_path)
no_synapses = #Add number of synapses
net = memory(no_synapses, synapse, R_thres)
t =  #Add retention_window
no_events = #Add total number of events
no_runs = #Add numbner of runs
lt_id =  #Add index of LTM consolidation start
intensities =  [] #Add how many times LTM is written
folders = ['{0:0>3}'.format(i) for i in range(len(intensities))]
signal_ids = [] #Add signal ids
for i, f in enumerate(folders):
    signal_path = os.path.join(data_path, f)
    net.plot_signal_palimpsest(signal_path, t, no_synapses, no_events, signal_ids)
    st_path = os.path.join(signal_path, 'st_lifetime.csv')
    net.plot_short_term_lifetime(st_path, t)
net.plot_short_term_avg_lifetime(data_path, folders, intensities)
