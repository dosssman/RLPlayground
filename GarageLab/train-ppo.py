import os
import gym
import argparse
import numpy as np
import tensorflow as tf

# Garage deps
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies.uniform_control_policy import UniformControlPolicy # Initial debug

# Custom deps
from algorithms.ppo import PPO
from replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from value_functions.value_function import VFunction
from value_functions.q_function import QFunction

# Less logging for TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
    # TODO: Arg Parser

    # Init env
    env = TfEnv( gym.make( "MountainCarContinuous-v0"))

    with tf.compat.v1.Session() as sess:

        # Initial Exploration Policy:
        initial_exploration_policy = UniformControlPolicy(
            env_spec = env.spec,
            # name='initial_exploration_policy'
        )

        # Placeholders
        o_ph = tf.placeholder( shape=[None] + list(env.spec.observation_space.shape),
            dtype=tf.float32, name='observations')
        n_o_ph = tf.placeholder( shape=[None] + list(env.spec.observation_space.shape),
            dtype=tf.float32, name='next_observations')
        a_ph = tf.placeholder( shape=[None] + list( env.spec.action_space.shape),
            dtype=tf.float32, name='actions')
        r_ph = tf.placeholder( shape=[None], dtype=tf.float32, name='rewards')
        ter_ph = tf.placeholder( shape=[None], dtype=tf.float32, name='terminals')

        # TODO: Pretiffy thos
        vf = VFunction( env.spec, o_ph)
        v = vf._vf

        with tf.variable_scope( 'target'):
            vf_targ = VFunction( env.spec, o_ph)
            v_targ = vf_targ._vf

        # Policy
        policy = GaussianMLPPolicy( env.spec)

        # Replay Buffer, TODO: Proper hyparametrization
        pool = SimpleReplayBuffer(
            env.spec,
            max_size = int( 1e5),
            batch_size = 128
        )

        # TODO: Implement sampler

        # Training loop

        # Buffer debug
        obs = env.reset()
        done = False
        current_timestep = 0

        while not done:
            # sampler.sample()
            # print( '# DEBUG: Current timstep %d' % current_timestep)
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step( action)
            transition = {
                'observation': obs,
                'action': action,
                'reward': reward,
                'terminal': done,
                'next_observation': next_obs
            }

            pool.add_transition( **transition)

            obs = next_obs
            current_timestep += 1

        print( 'sampling done:')
        print( 'size', pool.size)

        observations, actions, _, _, _ = pool.sample()
        print( 'sampled observatioons: ', observations.shape)

        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer())

            # obs_values = sess.run( v, feed_dict = { o_ph: observations})
            obs_values_targ = sess.run( v_targ, feed_dict = { o_ph: observations})

            # print( 'obs_values')
            # print( obs_values)
            print( 'v targs')
            print( obs_values_targ)
