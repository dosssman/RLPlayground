import numpy as np

class ReplayBuffer(object):
    def __init__( self, obs_dim, act_dim, size):
        self.obs_buf  = np.zeros( [size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros( [size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros( [size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros( size, dtype=np.float32)
        self.done_buf = np.zeros( size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store( self, o, a, r, o2, d):
        # Buffer must not be full
        assert self.ptr < self.max_size

        self.obs_buf[self.ptr] = o
        self.obs2_buf[self.ptr] = o2
        self.acts_buf[self.ptr] = a
        self.rews_buf[self.ptr] = r
        self.done_buf[self.ptr] = d
        self.ptr += 1
        self.size += 1
        
    def sample_batch( self, batch_size=32):
        idxs = np.random.randint( 0, self.size, size=batch_size)

        return dict( obs = self.obs_buf[idxs],
                     obs2 = self.obs2_buf[idxs],
                     rews = self.rews_buf[idxs],
                     acts = self.acts_buf[idxs],
                     done = self.done_buf[idxs])
