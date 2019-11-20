import numpy as np
import scipy.signal
from mpi_tools import mpi_statistics_scalar

class PPOBuffer:
	def __init__( self, obs_dim, act_dim, size, gamma=.99, lam=.95):
		self.obs_buf = np.zeros( shape=[size, obs_dim], dtype=np.float32)
		self.act_buf = np.zeros( shape=[size, act_dim], dtype=np.float32)
		self.adv_buf = np.zeros( size, dtype=np.float32)
		self.rew_buf = np.zeros( size, dtype=np.float32)
		self.ret_buf = np.zeros( size, dtype=np.float32)
		self.val_buf = np.zeros( size, dtype=np.float32)
		self.logp_buf = np.zeros( size, dtype=np.float32)
		self.gamma, self.lam = gamma, lam
		self.ptr, self.path_start_idx, self.max_size = 0, 0, size

	def store( self, obs, act, rew, val, logp):
		assert self.ptr < self.max_size
		self.obs_buf[self.ptr] = obs
		self.act_buf[self.ptr] = act
		self.rew_buf[self.ptr] = rew
		self.val_buf[self.ptr] = val
		self.logp_buf[self.ptr] = logp
		self.ptr += 1

	def finish_path( self, last_val=0):
		def discount_cumsum( x, discount):
			return scipy.signal.lfilter( [1], [1, float( -discount)], x[::-1], axis=0)[::-1]

		path_slice = slice( self.path_start_idx, self.ptr)
		rews = np.append( self.rew_buf[path_slice], last_val)
		vals = np.append( self.val_buf[path_slice], last_val)

		deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
		self.adv_buf[path_slice] = discount_cumsum( deltas, self.gamma * self.lam)
		self.ret_buf[path_slice] = discount_cumsum( rews, self.gamma)[:-1]

		self.path_start_idx = self.ptr

	def get( self):
		assert self.ptr == self.max_size
		self.ptr, self.path_start_idx = 0, 0
		adv_mean, adv_std = mpi_statistics_scalar( self.adv_buf)
		self.adv_buf = (self.adv_buf - adv_mean) / adv_std

		return [ self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]
