import numpy as np
import tensorflow as tf
import gym
import time
from logx import EpochLogger
# Buffer import
from buffer import GAEBuffer
# A2C model
from models import actor_critic as a2c
# Paralleization related
from mpi_tf import MpiAdamOptimizer, sync_all_params
from mpi_tools import *
EPS = 1e-8

def trpo( env_fn, actor_critic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000,
    epochs=50, gamma=.99, delta=.01, vf_lr=1e-3, train_v_iters=80,
    damping_coeff=.1, cg_iters=10, backtrack_iters=10, backtrack_coeff=.8,
    lam=.97, max_ep_len=1000, logger_kwargs=dict(), save_freq=10, algo="trpo"):

    # LOgger tools
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Seed inits
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Environment recreation
    env = env_fn()

    # Getting obs dims
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ac_kwargs['action_space'] = env.action_space

    # Placeholders
    x_ph, a_ph = tf.placeholder( name="x_ph", shape=[None, obs_dim], dtype=tf.float32), \
        tf.placeholder( name="a_ph", shape=[None, act_dim], dtype=tf.float32)
    adv_ph, ret_ph, logp_old_ph = tf.placeholder( name="adv_ph", shape=[None], dtype=tf.float32), \
        tf.placeholder( name="ret_ph", shape=[None], dtype=tf.float32), \
        tf.placeholder( name="logp_old_ph", shape=[None], dtype=tf.float32)

    pi, logp, logp_pi, info, info_phs, d_kl, v = actor_critic( x_ph, a_ph, **ac_kwargs)

    def keys_as_sorted_list(dict):
        return sorted(list(dict.keys()))

    def values_as_sorted_list(dict):
        return [dict[k] for k in keys_as_sorted_list(dict)]

    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph] + values_as_sorted_list(info_phs)

    get_action_ops = [pi, v, logp_pi] + values_as_sorted_list( info)

    # Experience buffer init
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    info_shapes = {k: v.shape.as_list()[1:] for k,v in info_phs.items()}
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    # Count variables
    def get_vars(scope=''):
        return [x for x in tf.trainable_variables() if scope in x.name]
    def count_vars(scope=''):
        v = get_vars(scope)
        return sum([np.prod(var.shape.as_list()) for var in v])

    var_counts = tuple( count_vars( scope) for scope in ["pi", "v"])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # TRPO Losses
    ratio = tf.exp( logp - logp_old_ph)
    pi_loss = -tf.reduce_mean( ratio * adv_ph)
    v_loss = tf.reduce_mean( (ret_ph - v)**2)

    # Optimizer for value function
    train_vf = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # CG solver requirements
    pi_params = get_vars( "pi")

    # Some helpers
    def flat_concat(xs):
        return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)
    def flat_grad(f, params):
        return flat_concat(tf.gradients(xs=params, ys=f))
    def hessian_vector_product(f, params):
        g = flat_grad(f, params)
        x = tf.placeholder(tf.float32, shape=g.shape)

        return x, flat_grad(tf.reduce_sum(g*x), params)
    def assign_params_from_flat(x, params):
        flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]

        return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

    gradient = flat_grad( pi_loss, pi_params)
    v_ph, hvp = hessian_vector_product(d_kl, pi_params)
    if damping_coeff > 0:
        hvp += damping_coeff * v_ph

    # Symbols for getting and setting params
    get_pi_params = flat_concat(pi_params)
    set_pi_params = assign_params_from_flat(v_ph, pi_params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def cg(Ax, b):
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_old = np.dot(r,r)

        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        # Prepare hessian func, gradient eval
        # Always so elegant haha
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        def mpi_avg(x):
            """Average a scalar or vector over MPI processes."""
            return mpi_sum(x) / num_procs()

        Hx = lambda x : mpi_avg(sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old, v_l_old = sess.run([gradient, pi_loss, v_loss], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2*delta/(np.dot(x, Hx(x))+EPS)) # OK
        old_params = sess.run(get_pi_params)

        def set_and_eval(step):
            sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})

            return mpi_avg(sess.run([d_kl, pi_loss], feed_dict=inputs))

        if algo=='npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)
        elif algo=="trpo":
            for j in range( backtrack_iters):
                kl, pi_l_new = set_and_eval( step=backtrack_coeff**j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.'%j)
                    logger.store(BacktrackIters=j)
                    break

                if j==backtrack_iters-1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function updates
        for _ in range(train_v_iters):
            sess.run(train_vf, feed_dict=inputs)
            v_l_new = sess.run(v_loss, feed_dict=inputs)

        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            agent_outs = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})
            a, v_t, logp_t, info_t = agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3:]

            # Save and log
            buf.store(o, a, r, v_t, logp_t, info_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

         # Perform TRPO or NPG update!
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
        logger.log_tabular('KL', average_only=True)
        if algo=='trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
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
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

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

    trpo(lambda : gym.make(args.env), actor_critic=a2c,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
