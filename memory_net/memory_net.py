import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
import pandas as pd
from copy import deepcopy
import pprint
from datetime import datetime
from scipy.stats import norm
import time
import os
import random
import sys
import glob

class memory(object):

    def __init__(self, no_synapses, synapse, R_thres):
        self.no_synapses = no_synapses
        self.synapse = {} #contains synaptic ensemble
        self.events = {} #contains memory events as values of each synapse
        self.time = [] #contains history of each synaptic value
        self.history = {} #resistance history
        self.weight = {} #synaptic weights
        self.signal = {} #memory signal to be tracked
        self.Vp = 1.4 #it was 8
        self.Vn = -2.6 #it was -5
        self.R_thres = R_thres
        self.trace_seed = None
        self.v_trace = [] #V for traced signal
        self.v_corr = {} #V for indexed correlated signals

        for i in range(no_synapses):
            self.synapse[i] = deepcopy(synapse)
            self.events[i] = {}
            self.events[i]["time"] = []
            self.events[i]["event"] = []

    def memory_reset(self):
        for i in self.synapse.keys():
            self.synapse[i].reset()
            self.events[i] = {}
            self.events[i]["time"] = []
            self.events[i]["event"] = []
            self.time = []
            self.signal = {}
            self.v_trace = []
            self.v_corr = {}
            self.trace_seed = None


    def remove_nan(self, data):
        #Return data array without NaN values
        nan_arr = np.isnan(data)
        not_nan_arr = ~ nan_arr
        data = np.array(data[not_nan_arr])
        return data

    def roullette_correlation(self, prob, trace=None, no_synapses=None, trace_similarity=None):
        #prob = probability for positive stimulation, i.e. potentiation
        #index traced signals, trace[0] would be the original and then trace[n] would be calculated via similarity
        #always pass on similarity variable
        size = (self.no_synapses if no_synapses==None else no_synapses)
        v = np.zeros(size)
        max_size = 2**32 - 1
        seed = random.randrange(max_size)

        if trace == 0:
            if self.trace_seed == None:
                self.trace_seed = seed #store the trace seed
            np.random.seed(self.trace_seed)
            draw = np.random.random(size=size)
            for j in range(draw.size):
                v[j] = (self.Vp if (draw[j]<prob) else self.Vn)
            self.v_corr[0] = v

        elif trace != None:
            if trace in self.v_corr.keys():
                v = self.v_corr[trace]
            else:
                np.random.seed(seed)
                draw = np.random.random(size=size)
                if trace_similarity != None:
                    for j in range(draw.size):
                        if draw[j] <= trace_similarity:
                            v[j] = self.v_corr[0][j]
                        else:
                            v[j] = (self.Vp if (self.v_corr[0][j] == self.Vn) else self.Vn)
                else:
                    for j in range(draw.size):
                        v[j] = (self.Vp if (draw[j]<prob) else self.Vn)
                self.v_corr[trace] = v

        else:
            np.random.seed(seed)
            draw = np.random.random(size=size)
            for j in range(draw.size):
                v[j] = (self.Vp if (draw[j]<prob) else self.Vn)

        return v

    def flip_signal(self, signal, signal_noise):
        #flip the signal; since only 1 voltage amplitude is used for each signal and the volatility model only observes the amplitude sign
        #the signal can be either 1 for potentiation or -1 for depression
        #signal_noise -> probability for signal to be inverted (e.g. potentiation to depression)
        draw = np.random.random()
        if draw < signal_noise:
            signal = (-1 if signal > 0 else 1)
        return signal

    def synapse_write_alloc(self, id, V, t, loop, no_events, noise, signal_noise=0, trace=False):
        #noise -> retention noise coming from volatility model
        #signal_noise -> probability for signal to be inverted (e.g. potentiation to depression)
        syn = self.synapse[id]
        no_values = int(t/syn.resolution + 1)
        """Important: write_speed set to 0"""
        # write_speed = syn.pw
        write_speed = 0
        start = int(loop*no_values)
        end = int((loop+1)*no_values)
        if signal_noise > 0:
            V = self.flip_signal(V, signal_noise)
        time, res = syn.retention(V, t, loop, noise=noise, cycle=True)
        w = [1 if res[i]<=self.R_thres else -1 for i in range(res.size)]
        if loop == 0:
            self.time = np.zeros(no_values*no_events)
            self.history[id] = np.zeros(no_values*no_events)
            self.weight[id] = np.zeros(no_values*no_events)
            self.events[id]["time"] = np.zeros(no_events)
            self.events[id]["event"] = np.zeros(no_events)
        else:
            time = time + write_speed + self.time[start-1]
        self.time[start:end] = time[:]
        self.history[id][start:end] = res[:]
        self.weight[id][start:end] = w[:]
        self.events[id]["time"][loop] = time[0]
        self.events[id]["event"][loop] = (1 if V>=0 else -1)
        if trace != False:
            self.signal[id] = (1 if V>=0 else -1)

    def correlation_stream(self, t, no_events, run, trace_index, trace_id, trace_intensity, trace_similarity=None, noise=True):
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
        prob = 0.5 #equally distributed potentiation/depression events
        trace_index = np.array(trace_index)
        trace_id = np.array(trace_id)
        trace_intensity = np.array(trace_intensity)
        loop = np.zeros(self.no_synapses)

        consolidate_runs = np.concatenate([np.arange(trace_index[i], trace_index[i]+trace_intensity[i]) for i in range(len(trace_index))])

        for i in range(no_events):
            if i%10== 0:
                t_start = datetime.now().replace(microsecond=0)
                print('Consolidation intensity: {}'.format(trace_intensity[0]))
                print('Synapses: {}, run {} of {}, event {} of {}'.format(self.no_synapses, run+1, self.no_runs, i+1, no_events))

            if i >= trace_index[0]: #start looking after the first trace has been written once
                passed_traces = trace_index[i >= trace_index]
                last_trace = passed_traces[-1]
                cycle = np.where(trace_index == last_trace)[0][0]

            in_cons = (i in consolidate_runs)
            trace = (trace_id[cycle] if in_cons else None)
            v = self.roullette_correlation(prob, trace=trace, trace_similarity=trace_similarity)

            for j, id in enumerate(self.synapse.keys()):
                self.synapse_write_alloc(id, v[j], t, int(loop[id]), no_events, noise, trace=trace)
                loop[id] += 1
            if i%10== 0:
                t_stop = datetime.now().replace(microsecond=0)
                elapsed = t_stop - t_start
                print('Time elapsed: {}'.format(elapsed))

    def correlation_stream_BCKP(self, t, no_events, noise, run, trace_index, trace_id, trace_intensity=None, trace_similarity=None):
        #Stream of multiple consolidated signals that have n% correlation to original trace
        #Here trace_id is an array indexing all signals to be traced
        prob = 0.5 #equally distributed potentiation/depression events
        trace_index = np.array(trace_index)
        trace_id = np.arange(trace_index.size)
        trace_intensity = np.array([trace_intensity for trace in trace_index])
        loop = np.zeros(self.no_synapses)
        consolidate_runs = np.concatenate([np.arange(trace_index[i], trace_index[i]+trace_intensity[i]) for i in range(len(trace_index))])
        for i in range(no_events):
            if i%10== 0:
                t_start = datetime.now().replace(microsecond=0)
                print('Consolidation intensity: {}'.format(trace_intensity[0]))
                print('Synapses: {}, run {} of {}, event {} of {}'.format(self.no_synapses, run+1, self.no_runs, i+1, no_events))
            if trace_index.all() != None:
                cycle = 0 #cycle = 0 before first trace index
                if i >= trace_index[0]:
                    #find the last event when a trace started
                    last_event = (trace_index[trace_index<=i][-1] if len(trace_index[trace_index<=i])>0 else trace_index[0])
                    cycle = np.where(trace_index == last_event)[0][0]
            else:
                cycle = (i // (consolidate_every) if i >= trace_start else 0) #find which consolidation cycle we are at
            in_cons = (i in consolidate_runs)
            trace = (trace_id[cycle] if in_cons else None)
            v = self.roullette_correlation(prob, trace=trace, trace_similarity=trace_similarity)

            for j, id in enumerate(self.synapse.keys()):
                self.synapse_write_alloc(id, v[j], t, int(loop[id]), no_events, noise, trace=trace)
                loop[id] += 1
            if i%10== 0:
                t_stop = datetime.now().replace(microsecond=0)
                elapsed = t_stop - t_start
                print('Time elapsed: {}'.format(elapsed))

    def correlation_stream_BCKPBCKP(self, t, no_events, noise, run, consolidate=None, trace_id=None, trace_similarity=None, trace_index=None, trace_intensity=None):
        #Stream of multiple consolidated signals that have n% correlation to original trace
        #Here trace_id is an array indexing all signals to be traced
        prob = 0.5 #equally distributed potentiation/depression events
        trace_index = np.array(trace_index)
        trace_id = np.arange(trace_index.size)
        trace_intensity = np.array([trace_intensity for trace in trace_index])
        loop = np.zeros(self.no_synapses)
        if trace_index.all() != None:
            consolidate_runs = np.concatenate([np.arange(trace_index[i], trace_index[i]+trace_intensity[i]) for i in range(len(trace_index))])
        else:
            consolidate_every = 100
            trace_start = 50 #where consolidation of trace starts
            consolidate_runs = np.arange(trace_start, trace_start+consolidate)
            consolidate_repeat = int((no_events-trace_start)/consolidate_every) #how many times there will be consolidation
            consolidate_runs = np.reshape(np.array([consolidate_runs + i*consolidate_every for i in range(consolidate_repeat)]),-1)
        for i in range(no_events):
            if i%10== 0:
                t_start = datetime.now().replace(microsecond=0)
                print('Consolidation intensity: {}'.format(trace_intensity[0]))
                print('Synapses: {}, run {} of {}, event {} of {}'.format(self.no_synapses, run+1, self.no_runs, i+1, no_events))
            if trace_index.all() != None:
                cycle = 0 #cycle = 0 before first trace index
                if i >= trace_index[0]:
                    #find the last event when a trace started
                    last_event = (trace_index[trace_index<=i][-1] if len(trace_index[trace_index<=i])>0 else trace_index[0])
                    cycle = np.where(trace_index == last_event)[0][0]
            else:
                cycle = (i // (consolidate_every) if i >= trace_start else 0) #find which consolidation cycle we are at
            in_cons = (i in consolidate_runs)
            trace = (trace_id[cycle] if in_cons else None)
            v = self.roullette_correlation(prob, trace=trace, trace_similarity=trace_similarity)

            for j, id in enumerate(self.synapse.keys()):
                self.synapse_write_alloc(id, v[j], t, int(loop[id]), no_events, noise, trace=trace)
                loop[id] += 1
            if i%10== 0:
                t_stop = datetime.now().replace(microsecond=0)
                elapsed = t_stop - t_start
                print('Time elapsed: {}'.format(elapsed))

    def signal_overlap(self, time=None, signal=None, weights=None):
        #Calculate the overlap between the traced signal and the memory state at t.
        #Discrete convolution between the two signals
        time = (self.time if time is None else time)
        signal = (self.signal if signal is None else signal)
        weights = (self.weight if weights is None else weights)
        t_start = datetime.now().replace(microsecond=0)
        time_size = time.size
        overlap = np.zeros(time_size)

        weights = np.array([weights[key] for key in weights.keys()]).transpose()
        signal = np.array([signal[key] for key in signal.keys()])

        for i in range(time_size):
            overlap[i] = np.sum(weights[i] == signal)

        t_stop = datetime.now().replace(microsecond=0)
        elapsed = t_stop - t_start
        print('Overlap calculation: {}'.format(elapsed))
        return overlap

    def plot_image_snaps(self, data_path, time_stamps):
        #plot images of memory state at specific time snapshots

        colors = mcolors.CSS4_COLORS
        tab_colors = mcolors.TABLEAU_COLORS

        cmp = LinearSegmentedColormap.from_list('custom',
                                                [(0, tab_colors['tab:orange']),
                                                (1, colors['purple'])], N=2)

        cmp_inv = LinearSegmentedColormap.from_list('custom',
                                                [(0, colors['purple']),
                                                (1, colors['orange'])], N=2)

        no_stamps = len(time_stamps)
        cols = np.ceil(no_stamps/4)
        data = pd.read_csv(data_path)
        time = np.array(data['time'])
        data = data.drop('time', axis=1)
        data = np.array(data)
        history = {time[i]: data[i,:] for i in range(time.size)}
        fig = plt.figure(figsize=(8,7))
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
        for i, t in enumerate(time_stamps):
            stamp = np.around(t-time_stamps[0], decimals=1)
            snap = history[t]
            snap = np.array([(1 if history[t][j]<self.R_thres else 0) for j in range(history[t].size)])
            snap = snap.reshape(100,100)
            ax = plt.subplot(4, cols, i+1)
            ax.imshow(snap, cmap=cmp_inv)
            ax.set_title(f'Time = {stamp}s')
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    #
    def application_experiment(self, signals_path, t, no_runs, signal_intensity, ltm_noise=[0], export_dir=None):
        #No random signals here. Load signals from arrays
        self.t = t
        self.no_events = no_events
        self.no_runs = no_runs
        now = datetime.now()
        time_tag = now.strftime("%m%d%y_%H%M%S")
        if export_dir:
            experiment_dir = os.path.join(export_dir, time_tag)
            os.mkdir(experiment_dir)
        no_events = np.sum(signal_intensity)
        if isinstance(ltm_noise, list) == False: #check if ltm_is inserted as a scalar - numpy will cause an issue
            ltm_noise = [ltm_noise]
        noise_dirs = []
        signals = np.array(pd.read_csv(signals_path)).reshape((100,100,-1))
        for n in ltm_noise:
            noise_dirs.append(f'ltm_noise_{n}')
            for run in range(self.no_runs):
                loop = 0
                self.memory_reset()
                for i in range(signals.shape[-1]):
                    signal = signals[:,:,i] // 255
                    signal = signal.astype(int)
                    signal = signal.reshape(-1)
                    signal = np.array([signal[i] if signal[i]==1 else -1 for i in range(signal.size)]) #regular signal
                    for j in range(signal_intensity[i]):
                        print('Run {} of {}, event {} of {}'.format(run+1, self.no_runs, loop+1, no_events))
                        signal_noise = (n if i==0 else 0) #add noise in the LTM signal only
                        for s, id in enumerate(self.synapse.keys()):
                            self.synapse_write_alloc(id, signal[s], t, loop, no_events, noise=True, signal_noise=n)
                        loop += 1

                if export_dir:
                    self.export_data('history', experiment_dir, noise_dirs[-1], run=run)
                    self.export_data('weights', experiment_dir, noise_dirs[-1], run=run)
                    self.export_data('events', experiment_dir, noise_dirs[-1], run=run)

        if export_dir:
            return experiment_dir, noise_dirs

    def correlation_experiment(self, t, no_events, no_runs, trace_index, trace_id, trace_intensity, trace_similarity=None, noise=True, export_dir=None):
        #Complete experiment to export .csv file with synapse history and signal overlaps
        self.t = t
        self.no_events = no_events
        self.no_runs = no_runs
        self.overlap = {}

        if export_dir:
            #Create run folders as 000, 001, 002, etc.
            i = 0
            while True:
                folder = '{0:0>3}'.format(i)
                if not os.path.exists(os.path.join(export_dir, folder)):
                    os.mkdir(os.path.join(export_dir, folder))
                    del i
                    break
                i += 1

        for run in range(no_runs):
            self.memory_reset()
            self.correlation_stream(t, no_events, run, trace_index, trace_id, trace_intensity, trace_similarity=trace_similarity, noise=noise)

            if export_dir:
                self.export_data('history', export_dir, folder, run=run)
                self.export_data('weights', export_dir, folder, run=run)
                self.export_data('events', export_dir, folder, run=run)

        if export_dir:
            readme = os.path.join(export_dir, folder, 'readme.txt')
            with open(readme, "a") as file:
                file.write(f'Events = {no_events}\nretention window = {t}\n{trace_index = }\n{trace_id = }\n{trace_intensity = }\n')
            return folder

    def plot_short_term_avg_lifetime(self, parent_path, folders, intensities):
        resolution = 0.1
        scan_window = 150
        fig, ax = plt.subplots(figsize=(5, 4))# height was 4.4
        fig.subplots_adjust(top=0.92, bottom=0.12, left=0.15, right=0.95)
        cmap = plt.get_cmap('Dark2')
        for i, f in enumerate(folders):
            intensity = intensities[i]
            folder = os.path.join(parent_path, f)
            target = 'st_lifetime.csv'
            path = os.path.join(folder, target)
            data = pd.read_csv(path, skiprows=0)
            st_life = self.remove_nan(np.array(data['ST life']))
            #average over the last scan_window samples
            avg_life = np.array([(np.mean(st_life[:j+1]) if j < scan_window else np.mean(st_life[j-(scan_window-1):j+1])) for j in range(st_life.size)])
            memories = np.arange(avg_life.size)
            non_zeros = avg_life != 0
            avg_life = avg_life[non_zeros]
            memories = memories[non_zeros]
            ax.plot(memories, avg_life, color=cmap(i), label='S = {}'.format(intensity))
        plt.yscale('log')
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='both', direction='in')
        ax.set_xlabel('Memory no.')
        ax.set_ylabel('Average STM lifetime (s)')

        plt.legend()
        plt.show()

    def short_term_lifetime(self, parent_path, t, no_synapses, no_events, no_runs, lt_id, intensity):
        #Calculate the total time the signal of a short term memory coming after lt_id remains higher than the lt signal.
        #Only calculate the retention window of each st signal per id.
        retention_window = t
        resolution = 0.1
        trace_ids = np.concatenate(([lt_id], np.arange(lt_id+intensity, no_events)))
        time, mean_overlaps, events_stamps, traces, weights = self.correlation_overlaps(parent_path, t, no_synapses, no_events, no_runs, trace_ids, return_overlaps=True) #get all run overlaps to show min/max per memory
        time = np.around(time, decimals=3)
        events_stamps = np.around(events_stamps, decimals=3)
        lt_signal, st_signals = mean_overlaps[0], mean_overlaps[1:]
        no_signals = st_signals[:,0].size
        del(mean_overlaps)
        evolution = {s: [] for s in range(no_signals)}
        total_lifetime = np.zeros(no_signals)
        ltm_overlap = np.zeros(no_signals) #store overlap between LTM and each STM
        state_overlap = np.zeros(no_signals) #store overlap between STM and state of system pre-write
        for i in range(no_signals):
            # print('ST: {}'.format(i+1+lt_id))
            t_start = np.around(events_stamps[i+1], decimals=3)
            t_stop = np.around((t_start + retention_window), decimals=3)
            start_id = np.where(time == t_start)[0][0]
            stop_id = np.where(time == t_stop)[0][0]
            memory_time = time[start_id:stop_id]
            evolution[i] = np.zeros_like(memory_time)
            lt = lt_signal[start_id:stop_id]
            st = st_signals[i, start_id:stop_id]
            st_indices = np.where(st > lt)[0]
            lt_indices = np.where(st <= lt)[0]
            evolution[i][st_indices] = 1
            evolution[i][lt_indices] = 0
            total_lifetime[i] = st_indices.size * resolution #each timestep is 0.1s
            ltm_overlap[i] = np.sum(traces[0] == traces[i+1])/no_synapses
            state_overlap[i] = np.sum(weights[start_id-1] == traces[i+1])/no_synapses
            no_stm = np.sum(total_lifetime == 0)

        life = pd.DataFrame()
        temp = pd.DataFrame({'ST life': total_lifetime})
        life = pd.concat([life, temp], ignore_index=False, axis=1)
        ev = pd.DataFrame.from_dict(evolution)
        life = pd.concat([life, ev], ignore_index=False, axis=1)
        tag = f'st_lifetime.csv'
        life_path = os.path.join(parent_path, tag)
        life.to_csv(life_path, index=None, header=True)

        readme = os.path.join(parent_path, 'readme.txt')
        with open(readme, "a") as file:
            file.write(f'ST lifetime for signals {lt_id + intensity} to {no_events}\n')

    def plot_short_term_lifetime(self, parent_path, retention_window):
        resolution = 0.1
        data = pd.read_csv(parent_path, skiprows=0)
        total_lifetime = self.remove_nan(np.array(data['ST life']))
        evolution = data.drop(columns='ST life')
        evolution = np.array([evolution[key] for key in evolution.keys()]).transpose()
        nans = np.isnan(evolution[:,0]) #remove nan entries
        no_nans = ~ nans
        no_nans_size = len(evolution[:][no_nans])
        evolution = evolution[:no_nans_size,:]
        evolution = np.flip(evolution, axis=0)
        no_points = evolution.shape[0] #how many life sample points exitst
        no_signals = evolution.shape[1]
        fig, axes = plt.subplots(figsize=(7, 4), nrows=1, ncols=2)# height was 4.4
        fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.90)
        #Subplot 1 --> Memory evolution
        ax1 = axes[0]
        ax1.imshow(evolution, aspect='auto', cmap=plt.get_cmap('coolwarm'))
        ax1.tick_params(axis='both', direction='in', which='both')
        ax1.set_yticks(np.linspace(evolution[:,0].size, 0, 5))
        ax1.set_yticklabels(np.linspace(0, retention_window, 5))
        ax1.set_xlabel('Memories after LTM')
        ax1.set_ylabel('Time (s)')

        # Subplot 2 --> Lifetime histogram
        ax2 = axes[1]
        nbins = np.arange(0, retention_window+resolution+0.001, 0.1)-resolution/2
        n, bins, patches = ax2.hist(total_lifetime, bins=nbins, rwidth=1, color=plt.get_cmap('Purples')(150))
        ax2.tick_params(axis='both', direction='in', which='both')
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax2.set_ylim((-2, 82))

        ax2.set_xlabel('STM total lifetime (s)')
        ax2.set_ylabel('Occurences')

        # Subplot 3 --> Lifetime statistics
        pdf = n / no_signals
        cdf = np.cumsum(pdf)
        ax3 = ax2.twinx()
        ax3.tick_params(axis='both', direction='in', which='both')
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.125))
        x_line = np.arange(0, retention_window+0.001, resolution)

        ax3.plot(x_line, pdf, 'b', label='PDF')
        ax3.plot(x_line, cdf, 'r', label='CDF')
        ax3.set_ylabel('Probability')
        ax3.legend()

        plt.show()

    def correlation_overlaps(self, parent_path, t, no_synapses, no_events, no_runs, trace_index, return_overlaps=False):
        #Monitor signal overlaps of multiple signals
        #trace_id --> index of traced signal
        retention_window = t
        no_signals = len(trace_index)
        signal_overlaps = {i: {} for i in range(no_signals)}
        for run in range(no_runs):
            print(f'Computing overlaps at run {run+1} of {no_runs}.')
            events_path = 'events_N_' + str(no_synapses) + '_run_' + str(run) + '_e_' + str(no_events) + '_t_' + str(retention_window) + '.csv'
            events_path = os.path.join(parent_path, events_path)
            weights_path = 'weights_N_' + str(no_synapses) + '_run_' + str(run) + '_e_' + str(no_events) + '_t_' + str(retention_window) + '.csv'
            weights_path = os.path.join(parent_path, weights_path)
            events = np.array(pd.read_csv(events_path))
            events_stamps = events[trace_index,0]
            trace_start = 1
            traces = np.array([events[id, trace_start:] for id in trace_index])
            weights = np.transpose(np.array(pd.read_csv(weights_path)))
            time = weights[0,:]
            weights = {i-1: weights[i,:] for i in range(1, weights[1:,0].size+1)} #signal_overlap function needs a dict
            for i, id in enumerate(trace_index):
                if run == 0:
                    signal_overlaps[i]['time'] = time
                    signal_overlaps[i]['mean signal'] = np.zeros_like(time)
                trace = {s: traces[i,s] for s in range(traces[i,:].size)}
                overlap = self.signal_overlap(time=time, signal=trace, weights=weights)/no_synapses
                signal_overlaps[i]['mean signal'] = signal_overlaps[i]['mean signal'] + overlap/no_runs
                signal_overlaps[i]['overlap ' + str(run)] = overlap

        if return_overlaps:
            mean_overlaps = np.array([signal_overlaps[i]['mean signal'] for i in signal_overlaps.keys()])
            weights = np.array([weights[key] for key in weights.keys()]).transpose()
            return time, mean_overlaps, events_stamps, traces, weights
        else:
            for i, id in enumerate(trace_index):
                file_name = 'signal_' + str(i) + '_N_' + str(no_synapses) + '_e_' + str(no_events) + '_t_' + str(retention_window) + '.csv'
                file_name = os.path.join(parent_path, file_name)
                df = pd.DataFrame.from_dict(signal_overlaps[i])
                df.to_csv(file_name, index=None, header=True)
    #
    def plot_image_ltm_overlap(self, parent_path, path_folders, t, no_synapses, no_events, signal_id, ltm_noise):
        #plot mean overlap of LTM or a signle ID for multiple LTM noise levels
        retention_window = t
        memory_window = t + self.synapse[0].pw
        cmap = plt.get_cmap('Dark2')
        fig, ax = plt.subplots(figsize=(5.5,3))
        fig.subplots_adjust(top=0.95, bottom=0.15, left=0.1, right=0.95)
        for i, f in enumerate(path_folders):
            target_path = 'signal_' + str(signal_id) + '_N_' + str(no_synapses) + '_e_' + str(no_events) + '_t_' + str(retention_window) + '.csv'
            path = os.path.join(parent_path, f, target_path)
            data = pd.read_csv(path, skiprows=0)
            data = np.transpose(np.array(data))
            time = data[0,:]
            self.mean_overlap = data[1,:]
            alpha = 1
            ax.plot(time, self.mean_overlap, alpha=alpha, label='noise: {}'.format(ltm_noise[i]), color=cmap(i))
        ax.set_ylim((0.65, 1.05))
        ax.tick_params(which='both', direction='in')
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xticklabels(['NLTM', 'NLTM', 'NLTM', '0', '10', '20'])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal overlap')
        ax.legend(loc='lower left')
        plt.show()

    def plot_signal_palimpsest(self, parent_path, t, no_synapses, no_events, signal_ids):
        #plot mean signals of overlapping memories
        noise = 50
        retention_window = t
        memory_window = t + self.synapse[0].pw
        noise_line = np.full(2, noise)
        signal_labels = np.concatenate((['LTM'], ['STM' + r'$_{}$'.format(i) for i in range(1, len(signal_ids))]))
        fig, ax = plt.subplots(figsize=(5,3.5))
        cmap = plt.get_cmap('Dark2')
        for i, s in enumerate(signal_ids):
            target_path = 'signal_' + str(s) + '_N_' + str(no_synapses) + '_e_' + str(no_events) + '_t_' + str(retention_window) + '.csv'
            path = os.path.join(parent_path, target_path)
            data = pd.read_csv(path, skiprows=0)
            data = np.transpose(np.array(data))
            time = data[0,:]
            time /= memory_window #quantise in events
            self.mean_overlap = data[1,:]
            alpha = (1 if s==0 else 0.7)
            ax.plot(time, self.mean_overlap*100, alpha=alpha, label=signal_labels[i], color=cmap(i))
        ax.plot([time[0], time[-1]], noise_line, 'k-', label='noise floor')
        plt.xlabel('Memory no.')
        ax.set_ylim((35, 105))
        ax.set_xlim((0, no_events))
        plt.ylabel('Signal overlap (%)')
        ax.tick_params(which='both', direction='in')
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(12.5))
        ax.legend()

        plt.show()

    def export_data(self, type, export_dir, tag, run=None):
        export_dir = os.path.join(export_dir, tag)
        exists = os.path.exists(export_dir)
        if not exists:
            os.mkdir(export_dir)

        if type == 'signal':
            runs = (self.no_runs if run is None else (run+1))
            mean_overlap = self.mean_overlap / runs
            export = pd.DataFrame()
            export['time'] = self.time
            export['mean signal'] = mean_overlap
            del(mean_overlap)
            for run in range(runs):
                temp = pd.DataFrame()
                label = 'overlap ' + str(run)
                temp[label] = self.overlap[run]
                export = pd.concat([export, temp], ignore_index=False, axis=1)
            export_path = 'signal_N_' + str(self.no_synapses) + '_e_' + str(self.no_events) + '_t_' +str(self.t) + '.csv'
            export_path = os.path.join(export_dir, export_path)
            export.to_csv(export_path, index=None, header=True)

        if type == 'events':
            export = pd.DataFrame()
            export["time"] = self.events[0]["time"]
            for id in self.synapse.keys():
                temp = pd.DataFrame()
                label = 'synapse ' + str(id)
                export[label] = self.events[id]["event"]
            export_path = 'events_N_' + str(self.no_synapses) + '_run_' + str(run) + '_e_' + str(self.no_events) + '_t_' +str(self.t) + '.csv'
            export_path = os.path.join(export_dir, export_path)
            export.to_csv(export_path, index=None, header=True)

        if type == 'weights':
            export = pd.DataFrame(self.weight)
            export.insert(0, 'time', self.time)
            labels = np.concatenate((['time'], np.array(['weight ' + str(id) for id in self.synapse.keys()])))
            export.columns = labels
            export_path = 'weights_N_' + str(self.no_synapses) + '_run_' + str(run) + '_e_' + str(self.no_events) + '_t_' +str(self.t) + '.csv'
            export_path = os.path.join(export_dir, export_path)
            export.to_csv(export_path, index=None, header=True)

        if type == 'history':
            export = pd.DataFrame(self.history)
            export.insert(0, 'time', self.time)
            labels = np.concatenate((['time'], np.array(['history ' + str(id) for id in self.synapse.keys()])))
            export.columns = labels
            export_path = 'history_N_' + str(self.no_synapses) + '_run_' + str(run) + '_e_' + str(self.no_events) + '_t_' +str(self.t) + '.csv'
            export_path = os.path.join(export_dir, export_path)
            export.to_csv(export_path, index=None, header=True)
