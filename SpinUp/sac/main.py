import gym
import numpy as np
import tensorflow as tf
import time

# Importing logger
from logx import EpochLogger

# Replay Buffer
from buffer import ReplayBuffer

def sac( env_fn, seed=0, gamma=.99, lam=.97, hidden_sizes=(200,100), alpha=.5,
    v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, polyak=1e-2, epochs=50, steps_per_epoch=1000,
    batch_size=100, start_steps=10000, logger_kwargs=dict(), replay_size=int(1e6),
    max_ep_len=1000, save_freq=1):

    logger = EpochLogger( **logger_kwargs)
    logger.save_config( locals())

    tf.set_random_seed(seed)
    np.random.seed( seed)

    env, test_env = env_fn(), env_fn()

    env = env_fn()

    # Dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_limit = env.action_space.high[0]

    # Placeholders
    x_ph = tf.placeholder( shape=[None,obs_dim], dtype=tf.float32)
    a_ph = tf.placeholder( shape=[None,act_dim], dtype=tf.float32)
    x2_ph = tf.placeholder( shape=[None,obs_dim], dtype=tf.float32)
    r_ph = tf.placeholder( shape=[None], dtype=tf.float32)
    d_ph = tf.placeholder( shape=[None], dtype=tf.float32)

    # Networks
    def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    # Why isn't the k used here ?
    def gaussian_likelihood(x, mu, log_std):
        EPS = 1e-8
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def clip_but_pass_gradient(x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
        act_dim = a.shape.as_list()[-1]
        net = mlp(x, list(hidden_sizes), activation, activation)
        mu = tf.layers.dense(net, act_dim, activation=output_activation)

        """
        Because algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To
        protect against that, we'll constrain the output range of the
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
        slightly different from the trick used by the original authors of
        SAC---they used tf.clip_by_value instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        return mu, pi, logp_pi

    def apply_squashing_func(mu, pi, logp_pi):
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
        return mu, pi, logp_pi

    with tf.variable_scope( "main"):
        activation = tf.tanh
        with tf.variable_scope("pi"):
            # mu = mlp( x_ph, hidden_sizes, activation, None)
            # log_std = mlp( mu, (act_dim,), activation, None)
            # # Avoid out of range log_std. Refer to Github for explanation.
            # log_std = LOG_STD_MIN + .5 * ( LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            #
            # mu = mlp( mu, (act_dim,), activation, None)
            #
            # pi = mu + tf.exp( log_std) * tf.random_normal( tf.shape(mu))
            # logp_pi = gaussian_likelihood( pi, mu, log_std)
            #
            # # Follow SpinningUp Implementation
            # mu = tf.tanh(mu)
            # pi = tf.tanh(pi)
            #
            # def clip_but_pass_gradient(x, l=-1., u=1.):
            #     clip_up = tf.cast(x > u, tf.float32)
            #     clip_low = tf.cast(x < l, tf.float32)
            #     # What is this supposed to mean even ?
            #     return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
            #
            # # Shameless copy paste
            # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)

            # Not working version bak
            # squashed_pi = tf.tanh( pi)
            #
            # # To be sure
            # pi = tf.clip_by_value( pi, -act_limit, act_limit)
            #
            # # Must take in the squased polic
            # log_squash_pi = gaussian_likelihood( squashed_pi, mu, log_std)

            # Shamefull plug
            mu, pi, logp_pi = mlp_gaussian_policy(x_ph, a_ph, hidden_sizes, tf.tanh, None)
            mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        with tf.variable_scope( "q1"):
            q1 = tf.squeeze(mlp( tf.concat( [x_ph, a_ph], -1), hidden_sizes+(1,),
                activation, None),axis=-1)

        with tf.variable_scope( "q1", reuse=True):
            q1_pi = tf.squeeze(mlp( tf.concat( [x_ph, pi], -1), hidden_sizes+(1,),
                activation, None ), axis=-1)


        with tf.variable_scope( "q2"):
            q2 = tf.squeeze( mlp( tf.concat( [x_ph, a_ph], -1), hidden_sizes+(1,),
                activation, None), axis=-1)

        with tf.variable_scope( "q2", reuse=True):
            q2_pi = tf.squeeze( mlp( tf.concat( [x_ph, pi], -1), hidden_sizes+(1,),
                activation, None ), axis=-1)

        with tf.variable_scope( "v"):
            # v = mlp( x_ph, hidden_sizes+(1,), activation, None)
            v = tf.squeeze( mlp( x_ph, hidden_sizes+(1,), activation, None), axis=-1)

    with tf.variable_scope( "target"):

        with tf.variable_scope( "v"):
            v_targ = tf.squeeze( mlp( x2_ph, hidden_sizes+(1,), activation, None), axis=-1)

    # helpers for var count
    def get_vars(scope=''):
        return [x for x in tf.trainable_variables() if scope in x.name]

    def count_vars(scope=''):
        v = get_vars(scope)
        return sum([np.prod(var.shape.as_list()) for var in v])

    # Count variables
    var_counts = tuple( count_vars( scope) for scope in ['main/pi',
        'main/q1', 'main/q2', 'main/v', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t v: %d, \t total: %d\n' % var_counts)

    # Targets
    q_backup_prestop = r_ph + gamma * (1-d_ph) * v_targ
    v_backup_prestop = tf.minimum( q1_pi, q2_pi) - alpha * logp_pi
    q_backup, v_backup = tf.stop_gradient( q_backup_prestop), tf.stop_gradient( v_backup_prestop)

    # Q Loss
    q1_loss = tf.reduce_mean( (q1 - q_backup) ** 2)
    q2_loss = tf.reduce_mean( (q2 - q_backup) ** 2)
    q_loss = q1_loss + q2_loss

    # V Loss
    v_loss = tf.reduce_mean( (v - v_backup) ** 2)

    # Pol loss
    pi_loss = tf.reduce_mean( - q1_pi + alpha * logp_pi)

    # Training ops
    v_trainop = tf.train.AdamOptimizer( v_lr).minimize( v_loss,
        var_list=tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="main/v"))
    q_trainop = tf.train.AdamOptimizer( q_lr).minimize( q_loss,
        var_list=tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="main/q"))
    pi_trainop = tf.train.AdamOptimizer( pi_lr).minimize( pi_loss,
        var_list=tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="main/pi"))

    assert polyak <= .5
    # Target update op
    init_v_target = tf.group( [ tf.assign( v_target, v_main)
        for v_main, v_target in zip(
            tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="main/v"),
            tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="target/v"))])

    update_v_target = tf.group( [ tf.assign( v_target, (1-polyak) * v_target + polyak * v_main)
        for v_main, v_target in zip(
            tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="main/v"),
            tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="target/v"))
    ])

    sess = tf.Session()
    sess.run( tf.global_variables_initializer())
    sess.run( init_v_target)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
        outputs={'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def test_agent( n=10):
        for j in range( n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0 ,0
            # print( o.reshape(-1, 1))
            # input()
            while not ( d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step( sess.run( pi, feed_dict={ x_ph: o.reshape(1, -1)}))
                ep_ret += r
                ep_len += 1

            logger.store( TestEpRet=ep_ret, TestEpLen=ep_len)

    #Buffer init
    buffer = ReplayBuffer( obs_dim, act_dim, replay_size)

    # Main loop
    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0 , 0
    total_steps = steps_per_epoch * epochs

    for t in range(total_steps):
        if t > start_steps:
            a = sess.run( pi, feed_dict={ x_ph: o.reshape(1, -1)})
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step( a)
        ep_ret += r
        ep_len += 1

        o2, r, d, _ = env.step( o)

        d = False or ( ep_len == max_ep_len)

        # Still needed ?
        o2 = np.squeeze( o2)

        buffer.store( o, a, r, o2, d)

        o = o2

        if d or ( ep_len == max_ep_len):
            for j in range( ep_len):
                batch = buffer.sample_batch( batch_size)
                feed_dict = {x_ph: batch['obs'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                # DEBUG:
                # v_backup_prestop_out = sess.run( v_backup_prestop, feed_dict=feed_dict)
                # print( v_backup_prestop_out.shape)
                # print( v_backup_prestop_out)
                # input()

                # Value gradient steps
                v_step_ops = [v_loss, v, v_trainop]
                outs = sess.run( v_step_ops, feed_dict)
                logger.store(LossV=outs[0], VVals=outs[1])

                # Q Gradient steps
                q_step_ops = [q_loss, q1, q2, q_trainop]
                outs = sess.run( q_step_ops, feed_dict)
                logger.store( LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                # Policy gradient steps
                # TODO Add entropy logging
                pi_step_ops = [ pi_loss, pi_trainop, update_v_target]
                outs = sess.run( pi_step_ops, feed_dict=feed_dict)
                logger.store( LossPi=outs[0])

            logger.store( EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0., 0

        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Saving the model
            if (epoch % save_freq == 0) or ( epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossV', average_only=True)
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
    parser.add_argument('--exp_name', type=str, default='sac')
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

    sac(lambda : gym.make(args.env),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
