import numpy as np
import tensorflow as tf
import gym
import time
from logx import EpochLogger
#
from buffer import ReplayBuffer
#
from models import actor_critic as a2c
# MPI SUpport
from mpi_tf import MpiAdamOptimizer, sync_all_params
from mpi_tools import *

# DDPG Function itself
def ddpg( env_fn, actor_critic=a2c, ac_kwargs=dict(), seed=0, steps_per_epoch=5000,
    epochs=100, replay_size=int(1e6), gamma=.99, polyak=.995, pi_lr=1e-3,
    q_lr=1e-3, batch_size=100, start_steps=10000, act_noise=.1, max_ep_len=1000,
    logger_kwargs=dict(), save_freq=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph, x2_ph, r_ph, d_ph = \
        tf.placeholder( name='x_ph', shape=[None, obs_dim], dtype=tf.float32), \
        tf.placeholder( name='a_ph', shape=[None, act_dim], dtype=tf.float32), \
        tf.placeholder( name='x2_ph', shape=[None, obs_dim], dtype=tf.float32), \
        tf.placeholder( name='r_ph', shape=[None], dtype=tf.float32), \
        tf.placeholder( name='d_ph', shape=[None], dtype=tf.float32)

    # Main networks
    with tf.variable_scope('main'):
        pi, q, q_pi = actor_critic( x_ph, a_ph, **ac_kwargs)

    # Target networks
    with tf.variable_scope('target'):
        pi_targ, _, q_pi_targ = actor_critic( x2_ph, a_ph, **ac_kwargs)

    replaybuffer = ReplayBuffer( obs_dim, act_dim, replay_size)

    # helpers for var count
    def get_vars(scope=''):
        return [x for x in tf.trainable_variables() if scope in x.name]

    def count_vars(scope=''):
        v = get_vars(scope)
        return sum([np.prod(var.shape.as_list()) for var in v])

    var_counts = tuple( count_vars( scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

     # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # Losses
    pi_loss = - tf.reduce_mean( q_pi)
    q_loss = tf.reduce_mean( (q-backup)**2)

    # Optimizer and train ops
    train_pi_op = tf.train.AdamOptimizer(pi_lr).minimize( pi_loss,
        var_list=get_vars('main/pi'))
    train_q_op = tf.train.AdamOptimizer(q_loss).minimize( q_loss,
        var_list=get_vars('main/q'))

    # Update target networks
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Init targets
    target_init = tf.group([tf.assign(v_targ, v_main)
        for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run( tf.global_variables_initializer())
    sess.run( target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
        outputs={'pi': pi, 'q': q})

    def get_actions( o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)

        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_actions(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop:
    for t in range( total_steps):
        if t > start_steps:
            a = get_actions( o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step( a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len==max_ep_len else d

        # Storing experience
        replaybuffer.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            for _ in range(ep_len):
                batch = replaybuffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }

                # Q-learning update
                outs = sess.run([q_loss, q, train_q_op], feed_dict)
                logger.store(LossQ=outs[0], QVals=outs[1])

                # Policy update
                outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            if( epoch % save_freq == 0) or (epoch == epochs -1):
                logger.save_state( {'env': env}, None)

            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    # Login related
    # Some fixed variable_scope
    import os.path as osp

    DEFAULT_DATA_DIR = osp.join(osp.abspath(
        osp.dirname(osp.dirname(__file__))),'data')
    FORCE_DATESTAMP = False
    DEFAULT_SHORTHAND = True
    WAIT_BEFORE_LAUNCH = 5

    def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):

        # Datestamp forcing
        datestamp = datestamp or FORCE_DATESTAMP

        # Make base path
        ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
        relpath = ''.join([ymd_time, exp_name])

        if seed is not None:
            # Make a seed-specific subfolder in the experiment directory.
            if datestamp:
                hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
            else:
                subfolder = ''.join([exp_name, '_s', str(seed)])
            relpath = osp.join(relpath, subfolder)

        data_dir = data_dir or DEFAULT_DATA_DIR
        logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                             exp_name=exp_name)
        return logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg(lambda : gym.make(args.env), actor_critic=a2c,
     ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
     gamma=args.gamma, seed=args.seed, epochs=args.epochs,
     logger_kwargs=logger_kwargs)
