import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import os

class operation(object):
    """
    Low level operation for simulated memristive synapses
    Mean model parameter values for Vp=1.4V and Vn=-2.6V
    """

    def __init__(self, R_init, R_bounds):

        self.R_bckp = R_init
        self.R_pre = R_init
        self.R = R_init #initialised R state
        self.R_low = R_bounds[0]
        self.R_high = R_bounds[1]

        #MODEL PARAMETERS
        self.alpha_p = -2130731.9
        self.alpha_n = 1166065.3
        self.tau_p = 0.728
        self.tau_n = 1.241
        self.beta_p = 0.537
        self.beta_n = 0.641
        self.offset_p = -514203.7
        self.offset_n = 512316.6

        self.pw = 500*510e-6 # (500 pulses 500μs pw, 10μs interpulse)
        self.mu_pos = 0.003 #positive noise distribution
        self.sigma_pos = 0.557
        self.mu_neg = 0.004 #negative noise distribution
        self.sigma_neg = 0.646
        self.resolution = 0.1 #retention sample resolution

    def reset(self):
        self.R_pre = self.R_bckp
        self.R = self.R_bckp

    def alpha(self, V):
        alpha = (self.alpha_p if V>0 else self.alpha_n)
        return alpha

    def tau(self, V):
        tau = (self.tau_p if V>0 else self.tau_n)
        return tau

    def beta(self, V):
        beta = (self.beta_p if V>0 else self.beta_n)
        return beta

    def offset(self, V, loop):
        offset = (self.offset_p if V>0 else self.offset_n)
        return offset

    def stretched_exp(self, V, t, alpha, tau, beta, gamma):

        temp = alpha * np.exp(-np.power((t/tau), beta)) + gamma
        R = np.zeros_like(temp)
        for i, r in enumerate(temp):
            if V > 0:
                R[i] = np.piecewise(temp[i], [temp[i] >= self.R_low, temp[i] < self.R_low], [temp[i], self.R_low])
            else:
                R[i] = np.piecewise(temp[i], [temp[i] <= self.R_high, temp[i] > self.R_high], [temp[i], self.R_high])
        return R

    def exp_decay(self, x, alpha, tau, gamma):
        return alpha * np.exp(-(x/tau)) + gamma

    def linear(self, x, slope, intercept):
        return slope*x + intercept

class memsynapse(operation):
    """
    Synaptic operations based on operation model class
    """

    def __init__(self, R_init, R_bounds):
        super().__init__(R_init, R_bounds)

    def make_staircase(self, min, max, step):
        #Create staircase parameter list
        staircase = []
        no_values = int((max - min)/step + 1)
        values = np.linspace(min, max, no_values)
        for i in range(values.size + 1):
            staircase.extend(values[:i])
        del(no_values, values)

        return np.array(staircase)

    def roullette(self, prob, size, random=True):
        #prob = probability for positive stimulation, i.e. potentiation
        v = np.zeros(size)
        if random:
            # now = int(time.time())
            seed = np.random.randint(1, 1e6)
            # print(seed)
            np.random.seed(seed)
        else:
            np.random.seed(123456)
        draw = np.random.random(size=size)
        for j in range(draw.size):
            v[j] = (self.Vp if (draw[j]<prob) else self.Vn)
        return v

    def volatility(self, V, t, loop, noise=True):
        alpha = self.alpha(V)
        tau = self.tau(V)
        beta = self.beta(V)
        offset = self.offset(V, loop)
        gamma = self.R_pre + offset

        res = self.stretched_exp(V, t, alpha, tau, beta, gamma)
        if noise:
            mu = (self.mu_pos if V>0 else self.mu_neg)
            sigma = (self.sigma_pos if V>0 else self.sigma_neg)
            shape = res.shape
            noise = np.random.normal(mu, sigma, shape)/100
            # print(noise)
            return (res + np.multiply(res, noise))
        else:
            return res

    def dR(self, V, loop):
        return self.volatility(V, 0, loop) - self.R_pre

    def retention(self, V, t, loop, noise=True, cycle=False, tolist=False):
        step = self.resolution
        no_values = int(t/step + 1)
        time = np.linspace(0, t, no_values)
        res = self.volatility(V, time, loop, noise=noise)
        if cycle == True:
            self.R_pre = res[-1]
            self.R = res[-1]

        if tolist:
            time = time.tolist()
            res = res.tolist()

        return time, res

    def stimulation_run(self, V, t, loops, R_thres, staircase=False, events=False, return_data=False):
        #staircase[min, max, step]
        time = []
        res = []
        if staircase and not events:
            V = self.make_staircase(staircase[0], staircase[1], staircase[2])
        elif events:
            V = np.array(events) * float(V)
        else:
            V = np.array([V])

        loop = 0
        first = True
        for i, v in enumerate(V):
            if i != 0:
                loop = (0 if np.sign(v) != np.sign(V[i-1]) else loop)
            for l in range(loops):
                # print('R_pre = {}'.format(self.R_pre))
                # print(loop)
                # print(v)
                _t, _r = self.retention(v, t, loop=loop, cycle=True, noise=False)
                if not first:
                    _t += time[-1]
                time.extend(_t)
                res.extend(_r)
                loop += 1
                first = False
        # print(V)
        time = np.array(time)
        res = np.array(res)

        if return_data:
            return time, res
        else:
            threshold_line = np.full(2, R_thres)
            threshold_axis = [time[0], time[-1]]
            plt.plot(time, res/1e6, alpha=0.7)
            plt.plot(threshold_axis, threshold_line/1e6, 'k-')
            plt.xlabel('Time (s)')
            plt.ylabel('Resistance (MΩ)')
            plt.title('Potentiation following LTD')
            plt.show()


    def write(self, N, V, R_low, R_high, return_value=False):

        self.R_pre = self.R
        dR = self.dR(N, V)
        R_temp = self.R + dR

        if V > 0:
            R_bound = R_low
            R_new = np.piecewise(R_temp, [R_temp > R_low, R_temp <= R_low], [R_temp, R_low])
        else:
            R_bound = R_high
            R_new = np.piecewise(R_temp, [R_temp < R_high, R_temp >= R_high], [R_temp, R_high])


        if return_value == False:
            self.R = R_new
        else:
            return R_new


    def finish(self, t, N, V, loop=0):
        R_end = self.volatility(t, N, V, loop)
        self.R_pre = R_end
        self.R = R_end


    def staircase(self, t, N, R_low, R_high, V, R_pre=None, export_res=False):
        #IF EXTERNAL RPRE IS USED ARRAY SIZE MUST BE EQUAL TO NO. LOOPS
        Rpre = {}
        time = {}
        res = {}
        R_start = {}
        loop = {}
        pulses = {}
        N = np.array(N)
        cnt = 0
        for i in range(N.size + 1):
            for j in range(i):
                n = N[j]
                pulses[cnt] = n

                try:
                    if Rpre != None:
                        self.R_pre = R_pre[cnt]
                except AttributeError:
                    pass

                if export_res == False:
                    r = self.R_pre
                    # print("N = {}, Rpre = {}".format(n, r))
                    if n not in time:
                        time[n] = {}
                        res[n] = {}
                        R_start[n] = {}
                        loop[n] = {}

                    R_start[n][r] = self.write(n, V, R_low, R_high, return_value=True)
                    loop[n][r] = cnt
                    time[n][r], res[n][r] = self.retention(t, n, V, loop=cnt, cycle=True)
                    cnt += 1
                else:
                    # Rpre[cnt] = self.R_pre
                    # print(V)
                    Rpre[cnt] = self.gamma(cnt, self.R_pre, V)

                    # print("Set: {}".format(self.R_pre))
                    time[cnt], res[cnt] = self.retention(t, n, V, loop=cnt, cycle=True)
                    # print("Sim: {}".format(self.R_pre))

                    cnt += 1
        print("done!")
        if export_res == False:
            return loop, R_start, time, res
        else:
            return time, res, Rpre

    def export_res(self, time, res, path):
        export = pd.DataFrame()
        for loop in time.keys():
            temp = pd.DataFrame()
            temp["loop"] = np.full(res[loop].size, loop)
            temp["time"] = time[loop]
            temp["res"] = res[loop]
            export = pd.concat([export, temp['loop'], temp['time'], temp['res']], ignore_index=False, axis=1)
        export.to_csv(path, index=None, header=True)
        print("Export complete!")

    def export_start_end(self, loop, R_start, res, path):
        # cnt = 0
        data = []
        for k1 in res.keys():
            for k2 in res[k1].keys():
                data.append(np.array([int(loop[k1][k2]), int(k1), k2, R_start[k1][k2], res[k1][k2][0], res[k1][k2][-1]]))
                # cnt += 1
        data = np.array(data)

        export = pd.DataFrame({"loop" : data[:,0],
                                "N" : data[:,1],
                                "R_pre" : data[:,2],
                                "R_start" : data[:,3],
                                "R_0" : data[:,4],
                                "R_end" : data[:,5]})

        export.to_csv(path, index=None, header=True)


# class device_test(device):
#
#     def __init__(self, R_init, R_bounds, loops):
#         super().__init__(R_init, R_bounds)
#         self.loops = loops
#
#     def test_param(self, param, V):
#
#         if param != "offset":
#             temp = np.zeros_like(V)
#             if param == "alpha":
#                 func = self.alpha
#             elif param == "tau":
#                 func = self.tau
#             elif param == "beta":
#                 func = self.beta
#             elif param == "dR":
#                 func = self.dR
#
#             try:
#                 for i, v in enumerate(V):
#                     temp[i] = func(v)
#             except TypeError:
#                 temp = [func(V)]
#
#         elif param == "offset":
#             func = self.offset
#             temp = np.zeros(self.loops)
#             for i in range(self.loops):
#                 try:
#                     temp[i] = func(V[0], i)
#                 except:
#                     temp[i] = func(V, i)
#         return temp
#
#     def plot_param(self, param, V):
#         if param != "gamma":
#             x_axis = self.V
#         else:
#             x_axis = np.arange(loops)
#
#         data = self.test_param(param, V)
#         plt.plot(x_axis, data, label="{} for {} V".format(param, V))
#         plt.legend()
#         plt.show()
#
#     def plot_parameters(self, V):
#
#         plt.figure()
#
#         plt.subplot(221)
#         param = 'alpha'
#         data = self.test_param(param, V)
#         plt.plot(V, data, 'bo')
#         plt.ylabel("alpha")
#         plt.xlabel("Amplitude (V)")
#         # plt.ylabel(param)
#
#         plt.subplot(222)
#         param = 'tau'
#         data = self.test_param(param, V)
#         plt.plot(V, data, 'bo')
#         plt.xlabel("Amplitude (V)")
#         plt.ylabel(param)
#
#         plt.subplot(223)
#         param = 'beta'
#         data = self.test_param(param, V)
#         plt.plot(V, data, 'bo')
#         plt.xlabel("Amplitude (V)")
#         plt.ylabel(param)
#
#         plt.subplot(224)
#         param = 'offset'
#         loops = np.arange(self.loops)
#         data = self.test_param(param, V)
#         plt.plot(loops, data)
#         plt.xlabel("Loop no.")
#         plt.ylabel("γ - Rpre")
#
#         plt.show()
#
#
#
# class parameters(object):
#
#     def __init__(self):
#         self.hidden = "hi"
#
#     def load(self, path):
#         df = pd.read_csv(path, usecols=[1])
#         return df.values.reshape(-1)
