import numpy as np
import tensorflow as tf

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def actor_critic( x, a, hidden_sizes=(400, 300), activation=tf.nn.relu,
    output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]

    with tf.variable_scope( 'pi'):
        # Just multiplying the last layers' output by act limit means what ? CLipping ?
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)

    with tf.variable_scope( 'q1'):
        q1 = tf.squeeze( mlp( tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    with tf.variable_scope( 'q2'):
        q2 = tf.squeeze( mlp( tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    with tf.variable_scope( 'q1', reuse=True):
        q1_pi = tf.squeeze( mlp( tf.concat( [x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    return pi, q1, q2, q1_pi

def actor_critic_pol_only( x, a, hidden_sizes=(400, 300), activation=tf.nn.relu,
    output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]

    with tf.variable_scope( 'pi'):
        # Just multiplying the last layers' output by act limit means what ? CLipping ?
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)

    return pi

def actor_critic_qvalues_only( x, a, hidden_sizes=(400, 300), activation=tf.nn.relu,
    output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]

    with tf.variable_scope( 'q1'):
        q1 = tf.squeeze( mlp( tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    with tf.variable_scope( 'q2'):
        q2 = tf.squeeze( mlp( tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    return q1, q2
