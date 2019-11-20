import numpy as np
import tensorflow as tf
from common import mlp

class QFunction( object):
    def __init__( self,
        env_spec,
        o_ph,
        a_ph,
        name="q_function",
        hidden_sizes=[32,32,],
        activation = tf.nn.relu,
        output_activation = None
        ):

        self._env_spec = env_spec
        self._name = name

        # TODO: Dynamically support disc. or cont action space
        with tf.variable_scope( self._name):
            x = tf.concat( [o_ph, a_ph], axis =1)

            x = mlp( x, hidden_sizes+[1], activation = activation,
                output_activation = output_activation)
            x = tf.squeeze( x)
            
        self._qf = x
