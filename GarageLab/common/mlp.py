import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.layers import dense

def mlp( x, hidden_sizes, activation = tf.nn.relu, output_activation = None):

    for units in hidden_sizes[:-1]:
        x = dense( x, units, activation = activation)

    return dense( x, hidden_sizes[-1], activation = output_activation)
