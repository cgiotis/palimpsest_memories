"""
Stream of multiple consolidated signals.
trace_index[] --> indices of first appearance for each signal that gets consolidated. MUST BE IN ASCENDING ORDER.
trace_id[] --> ids each consolidated signal, e.g. trace_id = [0,1,0,2] means that the first consolidated signal is again consolidated after the second.
trace_intensity[] --> consolidation strength for each traced signal.
trace_index, trace_id and trace_intensity must have the same shape.
Example: Out of 200 signals, consolidate the 10th signal twice, then the 40th signal 3 times and then rewrite the 10th signal 5 times
after 100 events:
    - trace_index = [9, 39, 100]
    - trace_id = [0, 1, 0]
    - trace_intensity = [2, 3, 5]
Optional: the user can choose whether the consolidated signals are n% similar to the first consolidated memory via trace_similarity.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from datetime import datetime
from synapse_model import memsynapse
from memory_net import memory

R_thres = 11e6
R_bounds = [1e3, 1e9]
synapse = memsynapse(R_thres, R_bounds)

"""
Correlation experiment; Construct a network with N identical synapses
subject to a random memory stream.
"""
now = datetime.now()
time_tag = now.strftime("%m%d%y_%H%M%S")
export_dir = '../data/network/random_memories/generated_data'
export_dir = os.path.join(export_dir, time_tag)
os.mkdir(export_dir)
no_synapses = #Add number of synapses
net = memory(no_synapses, synapse, R_thres)
t = #Add retention_window
no_events = #Add total number of events
no_runs = #Add numbner of runs
trace_index = []
trace_id = []
lt_intensity = []
for s in lt_intensity:
    trace_intensity = np.concatenate(([s], [1, 1, 1])) #consolidation strength
    """Initiate experiment"""
    folder = net.correlation_experiment(t, no_events, no_runs, trace_index, trace_id,\
                                        trace_intensity, export_dir=export_dir)
    """Calculate and save overlaps for traced signals"""
    data_path = os.path.join(export_dir, folder)
    net.correlation_overlaps(data_path, t, no_synapses, no_events, no_runs, trace_index)
    """
    Calculate the short-term lifetime of every memory after some signal lt_id that
    has been consolidated with some *intensity*
    """
    lt_id = trace_index[0]
    intensity = trace_intensity[0]
    net.short_term_lifetime(data_path, t, no_synapses, no_events, no_runs, lt_id, intensity)
