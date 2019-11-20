import numpy as np
import tensorflow as tf
from common import mlp

# TODO: Might transform to simple function
class VFunction( object):

    def __init__( self,
        env_spec,
        o_ph,
        hidden_sizes=[32,32,],
        activation = tf.nn.relu,
        output_activation = None,
        name='value_function',
        ):

        self._env_spec = env_spec
        self._name = name
        # Note: Do we neeed to create internal referemces for hidden_sizes,
        # activation, output_activation and such

        # Building the VF graph
        with tf.variable_scope( self._name):
            x = o_ph
            x = mlp( x, hidden_sizes+[1], activation = activation,
                output_activation = output_activation)
            x = tf.squeeze( x)

        # TODO: Unlegant, upgrade
        self._vf = x
