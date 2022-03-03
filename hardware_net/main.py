import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import chain
import hardware_net as network
import os

"""Plot device history"""
for bit in range(6):
    network.plot_history(bit)

"""Plot bit evolution"""
network.plot_bit_signal_evolution()

"""Plot signal evolution"""
network.plot_signal_evolution()

"""Plot recall probability"""
recall_percentage = [0.5, 0.8, 1]
no_trials = 200
network.plot_recall_probability(recall_percentage, no_trials)
