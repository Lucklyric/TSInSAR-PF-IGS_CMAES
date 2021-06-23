import numpy as np
from datetime import datetime
import re
import numpy as np
from numba import jit

@jit(nopython=True)
def wrap(phase):
    return np.angle(np.exp(1j * phase))


def get_delta_days(date_string):
    date_format = "%Y%m%d"
    tokens = re.split("_|\.", date_string)
    date1 = datetime.strptime(tokens[0], date_format)
    date2 = datetime.strptime(tokens[1], date_format)
    delta_days = np.abs((date2 - date1).days)
    return delta_days


def gen_sim_mr_he(num_of_mr, num_of_he, min_mr, max_mr, min_he, max_he):
    sim_signals = np.zeros([num_of_he * num_of_mr, 2])
    fmrs = np.random.uniform(min_mr, max_mr, num_of_mr).round(1)
    fhes = np.random.uniform(min_he, max_he, num_of_he).round(2)
    idx = 0
    for mr in fmrs:
        for he in fhes:
            sim_signals[idx] = [mr, he]
            idx += 1
    return sim_signals


# @jit(nopython=True)
def ri_l2(target, source, weight=1):
    # reward = np.abs(np.angle(1*np.exp(1j*(target-source)))) * weight
    reward = (np.square(np.sin(target) - np.sin(source)) * weight)
    reward += (np.square(np.cos(target) - np.cos(source)) * weight)
    reward = np.mean(reward)
    # if (isinstance(weight, int)):
    #     reward = np.mean(reward)
    # else:
    #     count = ((weight > 0).sum())
    #     reward = np.sum(reward)/count
    return reward

# @jit(nopython=True)
def ri_l1(target, source, weight=1):
    reward = (np.abs(np.sin(target) - np.sin(source)) * weight)
    reward += (np.abs(np.cos(target) - np.cos(source)) * weight)
    reward = np.mean(reward)
    # if (isinstance(weight, int)):
    #     reward = np.mean(reward)
    # else:
    #     count = ((weight > 0).sum())
    #     reward = np.sum(reward)/count
    return reward

def neg_coh(target, source, weight=1):
    diff = target - source 
    reward = np.abs(np.sum(np.exp(1j * (diff)))) / len(diff) 
    return 1-reward

def acc_diff_thresh(target, source, thresh):
    # percentage of phase phase diff below the threshold
    diff = target - source 
    acc = (np.abs(diff) < thresh).astype(int).mean()
    return np.abs(diff).mean(), acc

