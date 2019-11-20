import numpy as np
import tensorflow as tf

EPS = 1e-8
# Following SPinningUp of OpenAI
def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def gaussian_likelihood(x, mu, log_std):
    # Checkout this formulae
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))

    return tf.reduce_sum(pre_sum, axis=1)

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + EPS) - 1) +  log_std1 - log_std0

    all_kls = tf.reduce_sum( pre_sum, axis=1)
    return tf.reduce_mean( all_kls)

def mlp_gaussian_policy( x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable( name="log_std", initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp( log_std)
    pi = mu +  tf.random_normal( tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    old_mu_ph, old_log_std_ph = tf.placeholder( name="old_mu_ph", shape=[None, act_dim], dtype=tf.float32), \
        tf.placeholder( name="old_log_std_ph", shape=[None, act_dim], dtype=tf.float32)
    d_kl = diagonal_gaussian_kl( mu, log_std, old_mu_ph, old_log_std_ph)

    info = {'mu': mu, 'log_std': log_std}
    info_phs = {'mu': old_mu_ph, 'log_std': old_log_std_ph}

    return pi, logp, logp_pi, info, info_phs, d_kl

def actor_critic( x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):
    # Fixed to Guassian Policy
    with tf.variable_scope('pi'):
        pi, logp, logp_pi, info, info_phs, d_kl = mlp_gaussian_policy( x, a, hidden_sizes, activation,
            output_activation, action_space)

    with tf.variable_scope('v'):
        v = tf.squeeze( mlp( x, list(hidden_sizes)+[1], activation, None), axis=1)

    return pi, logp, logp_pi, info, info_phs, d_kl, v
