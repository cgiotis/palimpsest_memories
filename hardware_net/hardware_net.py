import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from itertools import chain, combinations
from scipy import stats
import os

PARENT_PATH = "../data/BITS"
BINARY_THRESHOLD = 10.6e6

RET_INIT_TAG = "init.csv"
RET_TAG = "ret.csv"
FIT_TAG = "fit.csv"
STIMULI_TAG = "stimuli.csv"

MU = 0.02 #noise distribution mean
SIGMA = 1.24 #noise distribution std. deviation

BITS = np.arange(6)
ORIGINAL_SIGNAL = np.array([1,0,1,1,0,0])
RETENTION_MAX = 29.9 #maximum retention window (seconds)
RESOLUTION = 0.1 #plotting resolution (seconds)

def remove_nan(data):
    #Return data array without NaN values
    nan_arr = np.isnan(data)
    not_nan_arr = ~ nan_arr
    data = np.array(data[not_nan_arr])
    return data

def cast_to_range(value, min, max, range_min, range_max):
    n = ((value-min)/(max-min)) * (range_max-range_min) + range_min
    return n

def load_retention(bit, data_type, retentions=None, add_up=True, concatenate=False, return_path=False):
    #Retention file has 1 column with N(pulse no.) history and then follows a time_x, res_x pattern.
    if data_type == 'init':
        data_tag = RET_INIT_TAG
    elif data_type == 'ret':
        data_tag = RET_TAG
    elif data_type == 'fit':
        data_tag = FIT_TAG
    else:
        raise TypeError

    file_name = f'bit_{bit}_{data_tag}'
    data_path = os.path.join(PARENT_PATH, file_name)

    if retentions!=None:
        cols = list(chain.from_iterable((2*i+1, 2*i+2) for i in retentions))
        data = pd.read_csv(data_path, usecols=cols, skiprows=0)
    else:
        data = pd.read_csv(data_path, skiprows=0)
        data = data.drop(data.columns[0], axis=1)
    data = np.transpose(np.array(data))

    for i in range(2, data[:,0].size-1, 2):
        #Accumulate time domain by addding the last non Nan retention value to the next one.
        last = 1
        while(np.isnan(data[i-2,-last])):
            last += 1
        if add_up == True:
            data[i,:] += data[i-2,-last]

    time = []
    res = []

    for i in range(int(data[:,0].size/2)):
        if concatenate:
            nan_arr = np.isnan(data[2*i,:])
            not_nan_arr = ~ nan_arr
            time.extend(np.array(data[2*i,not_nan_arr]))
            res.extend(np.array(data[2*i+1,not_nan_arr]))
        else:
            time = data[0::2, :]
            res = data[1::2, :]

    time = np.array(time)
    res = np.array(res)

    if return_path:
        return time, res, data_path
    else:
        return time, res

def plot_history(bit):
    #Plot synapse's history along with initialised binary threshold.
    colors = mcolors.CSS4_COLORS
    res_color = colors['purple']
    phase0_color = colors['bisque']
    phase1_color = colors['darkorange']
    phase2_color = colors['mediumpurple']
    write_color = 'firebrick'
    read_color = 'darkblue'
    marker_size = 2.5
    file_name = f'bit_{bit}_{STIMULI_TAG}'
    data_path = os.path.join(PARENT_PATH, file_name)
    data = pd.read_csv(data_path, usecols=(1,2,3,4,5), skiprows=0)
    data = np.transpose(np.array(data))

    res = remove_nan(data[0,:])
    res_min = np.amin(res)/1e6
    res_max = np.amax(res)/1e6
    res_indices = np.arange(res.size)

    read_index = remove_nan(data[1,:])
    read_pulses = remove_nan(data[2,:])
    write_index = remove_nan(data[3,:])
    write_pulses = remove_nan(data[4,:])
    write_min = np.amin(write_pulses)
    write_max = np.amax(write_pulses)

    xpad = 1500
    res_ypad = 1
    pulses_ypad = 2

    fig = plt.figure(figsize=(6,4), constrained_layout=True)
    gs = fig.add_gridspec(3,1)
    p1 = fig.add_subplot(gs[:2,:])
    p1.plot(res_indices, res/1e6, linestyle='-', color=res_color, alpha=0.5)
    p1.plot(res_indices, res/1e6, linestyle='None', marker='o', markersize=marker_size, mfc=res_color, mec=res_color, label='bit: {}'.format(bit))
    threshold_line = p1.axhline(y=BINARY_THRESHOLD/1e6, xmin=res_indices[0]-xpad, xmax=res_indices[-1]+xpad)
    plt.setp(threshold_line, linestyle='dashed', color='k', alpha=0.6, label='Mean threshold')
    p1.axvspan(-500, res_indices[3646], color=phase0_color, ec=None, alpha=0.3)
    p1.axvspan(res_indices[3647], res_indices[27285], color=phase1_color, ec=None, alpha=0.2)
    p1.axvspan(res_indices[27286], res_indices[-1]+500, color=phase2_color, ec=None, alpha=0.3)
    p1.axis([res_indices[0]-xpad, res_indices[-1]+xpad, res_min-res_ypad, res_max+res_ypad])
    p1.tick_params(direction='in')
    plt.ylabel('Resistance (MΩ)')

    p2 = fig.add_subplot(gs[-1:,:])
    zero_line = p2.axhline(y=0, xmin=res_indices[0]-xpad, xmax=res_indices[-1]+xpad)
    plt.setp(zero_line, color='black')
    markerline, stemlines, baseline = p2.stem(write_index, write_pulses, use_line_collection=True, label='write')
    read = p2.plot(read_index, read_pulses, linestyle='None', marker='o', markersize=2, color=read_color, label='read')
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, color=write_color, alpha=0.5)
    plt.setp(markerline, marker='o', markersize=3, color=write_color)
    plt.ylabel('Amplitude (V)')
    plt.xlabel('Pulse no.')
    p2.tick_params(direction='in')
    p2.axis([res_indices[0]-xpad, res_indices[-1]+xpad, write_min-pulses_ypad, write_max+pulses_ypad])
    p2.legend()
    plt.show()

def adds_noise(data):
    #Adds noise on fitted R data, where noise is drawn from a normal distribution
    #and its value adds a factored changes s.t. R_noise = R + R*noise/100
    size = data.size
    noise = np.random.normal(MU, SIGMA, size)
    data += data * noise/100
    return data

def signal_evolution(data_type):
    #Calculate the network's (%) overlap to the ORIGINAL_SIGNAL over time
    signal_size = ORIGINAL_SIGNAL.size
    add_noise = (True if data_type=='fit' else False)
    time, signal, no_steps = bit_signal_evolution(data_type, add_noise=add_noise, bitwise=True, return_no_steps=True)
    no_events = int(time.size / no_steps) #no. timestamps per stimulation
    signal_overlap = np.zeros_like(time)
    overlap = {}

    for i, bit in enumerate(signal.keys()):
        bit_overlap = np.zeros_like(signal[bit])
        for j in range(bit_overlap.size):
            s = signal[bit][j]
            bit_overlap[j] = np.piecewise(s, [s==ORIGINAL_SIGNAL[i], s!=ORIGINAL_SIGNAL[i]], [1, 0])
        overlap[i] = bit_overlap

    for i in range(signal_size):
        signal_overlap += overlap[i]
    signal_overlap = 100 * (signal_overlap/signal_size)

    return signal_overlap, time, no_events

def plot_signal_evolution():
    #Plot (%) overlap between signal and original
    data_type = 'ret'
    time_steps = np.arange(0, RETENTION_MAX, RESOLUTION)
    no_steps = time_steps.size
    del(time_steps)

    signal_overlap, time, no_events = signal_evolution(data_type)
    checkpoint = 2*no_steps
    box = dict(boxstyle='square', ec='k', fc='w', alpha=1)
    plt.figure(figsize=(5, 3.5))

    p1 = plt.plot(time, signal_overlap, linestyle='None', marker='o', markersize=1, color='purple')
    p2 = plt.plot(time, signal_overlap, linestyle='-', color='purple', alpha=0.5, label='Signal overlap to $V_1$')

    plt.tick_params(direction='in')
    plt.xticks(labels=None)
    plt.yticks(labels=None)
    plt.ylabel('Signal Overlap (%)')
    plt.xlabel('Time (s)')
    plt.show()

def bit_signal_evolution(ret_type, add_noise=False, bitwise=True, return_no_steps=False):
    #Examine R(t) for each synapse and record the corresponding states over
    #time = np.linspace(0, RETENTION_MAX, RESOLUTION)
    #signal[key] has the same shape for all keys.
    data_type = 'ret'
    signal = {}
    time_steps = np.arange(0, RETENTION_MAX, RESOLUTION)
    no_steps = time_steps.size
    for bit in BITS:
        time_total = []
        offset = 0
        signal[bit] = []
        time, res = load_retention(bit, data_type, add_up=False, concatenate=False)
        no_loops = time[:,0].size
        for loop in range(no_loops):
            time_total.extend(time_steps + offset)
            for i, step in enumerate(time_steps):
                indices = np.reshape(np.nonzero(time[loop,:] >= step), -1)
                _r = res[loop, indices[0]]
                if add_noise:
                    _r = adds_noise(_r)
                if bitwise:
                    _s = np.piecewise(_r, [_r >= BINARY_THRESHOLD, _r < BINARY_THRESHOLD], [0, 1])
                    signal[bit].append(int(_s))
                else:
                    signal[bit].append(float(_r))
            offset = time_total[-1]

        signal[bit] = np.array(signal[bit])
    time_total = np.array(time_total)

    if return_no_steps:
        return time_total, signal, no_steps
    else:
        return time_total, signal

def plot_bit_signal_evolution():
    #Produce a heatmap representation of the state evolution for each bit
    data_type = 'ret' #load raw retention data
    time, signal, no_steps = bit_signal_evolution(data_type, bitwise=False, return_no_steps=True)
    no_points = time.size
    no_events = int(time.size / no_steps)
    checkpoint = 3*no_steps

    values = np.reshape(np.vstack([signal[key][-checkpoint:] for key in signal.keys()]), -1)
    vmin, vmax = np.amin(values), np.amax(values)

    events = np.full_like(time, BINARY_THRESHOLD)
    pot = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]

    n = BINARY_THRESHOLD
    n = cast_to_range(n, vmin, vmax, 0, 1)
    colors = mcolors.CSS4_COLORS
    tab_colors = mcolors.TABLEAU_COLORS


    cmp = LinearSegmentedColormap.from_list('custom',
                                            [(0, tab_colors['tab:orange']),
                                            (0.2, colors['orange']),
                                            (n, colors['mediumpurple']),
                                            (1, colors['purple'])], N=10)

    n_rows = len(signal.keys())
    fig_height = 0.35 + 0.15 + (n_rows + (n_rows-1)*0.1)*0.22
    fig, axes = plt.subplots(nrows=n_rows, figsize=(8, 2.5))

    fig.subplots_adjust(top=1-.15/fig_height, bottom=.35/fig_height, left=0.05, right=0.89)

    for i, key in enumerate(signal.keys()):
        s = signal[key]
        events = s
        for j in range(len(pot)):
            events[j*no_steps] = np.piecewise(pot[j], [pot[j]==1, pot[j]==-1], [vmin, vmax])
        events = np.vstack((events, events))
        bit = np.vstack((s, s))

        try:
            ax = axes[i]
        except TypeError:
            ax = axes

        im = ax.imshow(bit, vmin=vmin, vmax=vmax, aspect='auto', cmap=cmp)
        ax.set_yticks([0.5])
        ax.set_yticklabels([f'b{i}'])

        if i != 5:
            ax.set_xticklabels([])
            ax.set_xticks([])
        else:
            time = np.around(time, decimals=2)
            ticks = [0, 125, 250, 375, 500]
            xticks = [np.reshape(np.where(time == float(tick)), -1)[0] for tick in ticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(ticks)
            ax.set_xlabel('Time (s)')

    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, ticks=[vmin, BINARY_THRESHOLD, vmax])
    plt.show()

def recall_probability(recall_percentage, no_trials):
    #Calculate the probability of recalling at least recall_percentage (%) of the ORIGINAL_SIGNAL
    #over time - average over number of trials.
    data_type = 'fit'
    overlaps = {}

    for run in range(no_trials):
        print('Run {} of {}.'.format(run+1, no_trials))
        signal_overlap, time, no_events = signal_evolution(data_type)
        if run == 0:
            for t in time:
                overlaps[t] = np.zeros(no_trials)
            recall_prob = []
        for i, t in enumerate(overlaps.keys()):
            overlaps[t][run] = signal_overlap[i]/100

    for i, t in enumerate(overlaps.keys()):
        recall_count = np.reshape(np.where(overlaps[t] >= recall_percentage), -1)
        recall_prob.append(float(recall_count.size / no_trials))

    recall_prob = np.array(recall_prob)
    time = time[:recall_prob.size]

    return recall_prob, time, no_events

def plot_recall_probability(recall_percentage, no_trials):
    #Plot the recall probability, as described above
    colors = mcolors.CSS4_COLORS
    perc_colors = [colors['plum'], colors['mediumpurple'], colors['purple']]

    time_steps = np.arange(0, RETENTION_MAX, RESOLUTION)
    no_steps = time_steps.size
    del(time_steps)

    fig = plt.figure(figsize=(5, 3.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 100)
    ax = fig.add_subplot(gs[:,:])

    for i, perc in enumerate(recall_percentage):
        print('Calculating {} from {}'.format(perc, recall_percentage))
        color = perc_colors[i]
        recall_prob, time, no_events = recall_probability(perc, no_trials)
        recall_prob_zoom = recall_prob[-3*no_steps:]*100
        time_zoom = time[-3*no_steps:]
        ax.plot(time_zoom, recall_prob_zoom, linestyle='-', color=color, alpha=0.5, label='{} %'.format(int(perc*100)))
        ax.plot(time_zoom, recall_prob_zoom, linestyle='None', marker='o', markersize=1, color=color)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Recall probability')
    ax.legend(loc='lower left', bbox_to_anchor=(0.125, 0.05, 0.4, 0.2), fontsize=9.5, title='Partial $S_1$')
    ax.tick_params(axis='both', direction='in')
    ax.set_yticks([0, 50, 100])
    plt.show()
