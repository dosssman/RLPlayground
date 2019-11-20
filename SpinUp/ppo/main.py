import numpy as np
import tensorflow as tf
import gym
import time
from logx import EpochLogger
# Buffer import
from ppobuffer import PPOBuffer
# A2C Model
from models import actor_critic as a2c
# MPI SUpport
from mpi_tf import MpiAdamOptimizer, sync_all_params
from mpi_tools import *
# Anti NaN
EPS = 1e-8

def ppo( env_fn, actor_critic=a2c, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
    epochs=50, gamma=.99, clip_ratio=.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80,
    train_v_iters=80, lam=.97, max_ep_len=1000, target_kl=.01, logger_kwargs=dict(),
    save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Share action space structure with the actor_critic
    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph = tf.placeholder( name="x_ph", shape=[None, obs_dim], dtype=tf.float32), \
        tf.placeholder( name="a_ph", shape=[None, act_dim], dtype=tf.float32)
    adv_ph, ret_ph, logp_old_ph = tf.placeholder( name="adv_ph", shape=[None], dtype=tf.float32), \
        tf.placeholder( name="ret_ph", shape=[None], dtype=tf.float32), \
        tf.placeholder( name="logp_old_ph", shape=[None], dtype=tf.float32)

    # Main outputs from computation graph
    # print( actor_critic( x_ph, a_ph, **ac_kwargs))
    pi, logp, logp_pi, v = actor_critic( x_ph, a_ph, **ac_kwargs)

    all_phs = [ x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    get_action_ops = [ pi, v, logp_pi]

    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # helpers for var count
    def get_vars(scope=''):
        return [x for x in tf.trainable_variables() if scope in x.name]

    def count_vars(scope=''):
        v = get_vars(scope)
        return sum([np.prod(var.shape.as_list()) for var in v])

    var_counts = tuple(count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO Objectives
    ratio = tf.exp( logp - logp_old_ph)
    min_adv = tf.where( adv_ph >0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean( tf.minimum( ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Stats to watch
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)

    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    train_pi = MpiAdamOptimizer( learning_rate=pi_lr).minimize( pi_loss)
    train_v = MpiAdamOptimizer( learning_rate=vf_lr).minimize( v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})


    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        for i in range( train_pi_iters):
            _, kl = sess.run( [train_pi, approx_kl], feed_dict=inputs)
            def mpi_avg(x):
                """Average a scalar or vector over MPI processes."""
                return mpi_sum(x) / num_procs()
            kl = mpi_avg( kl)

            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break

        logger.store(StopIter=i)
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)

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

    ppo( lambda: gym.make( args.env), actor_critic=a2c,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
