import lasagne
import theano.tensor as T
import numpy as np
import theano
from lasagne.layers import Layer

class DummyLayer(Layer):

    def __init__(self, incoming, **kwargs):
    super(DummyLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        return input