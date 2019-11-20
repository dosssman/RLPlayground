import numpy as np
import scipy.signal
import tensorflow as tf
from mpi_tools import mpi_statistics_scalar

class GAEBuffer(object):
    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=.99, lam=.95):
        self.obs_buf = np.zeros( [size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros( [size, act_dim], dtype=np.float32)
        self.adv_buf = np.zeros( size, dtype=np.float32)
        self.rew_buf = np.zeros( size, dtype=np.float32)
        self.ret_buf = np.zeros( size, dtype=np.float32)
        self.val_buf = np.zeros( size, dtype=np.float32)
        self.logp_buf = np.zeros( size, dtype=np.float32)
        # What exactly happens here => Check info shape strucutre
        self.info_buffs = { k: np.zeros( [size] + list(v), dtype=np.float32)
            for k,v in info_shapes.items()}
        self.sorted_info_keys = sorted( list( self.info_buffs.keys()))
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store( self, obs, act, rew, val, logp, info):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        # Seems like Gym env's info data
        for i,k in enumerate( self.sorted_info_keys):
            self.info_buffs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path( self, last_val=0):
        def discount_cumsum( x, discount):
            return scipy.signal.lfilter( [1], [1, float( -discount)], x[::-1], axis=0)[::-1]

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get( self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0,0
        adv_mean, adv_std = mpi_statistics_scalar( self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        def keys_as_sorted_list(dict):
            return sorted(list(dict.keys()))

        def values_as_sorted_list(dict):
            return [dict[k] for k in keys_as_sorted_list(dict)]

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf] \
            + values_as_sorted_list(self.info_buffs)
