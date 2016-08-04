import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import ReshapeLayer, DenseLayer, InputLayer, TransformerLayer, ScaleLayer

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer

    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayer
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer

    print('Using lasagne.layers (slower)')


class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """

    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


# This builds a model of Conv. Autoencoder
def build_cnnae_network(input_shape):
    conv_filters = 32
    filter_size = 5
    pool_size = 2
    encode_size = input_shape[2] * 2

    network = InputLayer(shape=(None,
                                input_shape[1],
                                input_shape[2],
                                input_shape[3]))

    network = Conv2DLayer(network,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None)

    network = MaxPool2DLayer(network,
                             pool_size=(pool_size, pool_size))

    network = Conv2DLayer(network,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None)

    network = MaxPool2DLayer(network,
                             pool_size=(pool_size, pool_size))

    network = ReshapeLayer(network,
                           shape=([0], -1))

    network = DenseLayer(network,
                         name='encode',
                         num_units=encode_size)

    network = DenseLayer(network,
                         num_units=conv_filters * (input_shape[2] + filter_size - 1) ** 2 / 4)

    network = ReshapeLayer(network,
                           shape=([0],
                                  conv_filters,
                                  (input_shape[2] + filter_size - 1) / 2,
                                  (input_shape[2] + filter_size - 1) / 2))

    network = Unpool2DLayer(network,
                            ds=(pool_size, pool_size))

    network = Conv2DLayer(network,
                          num_filters=input_shape[1],
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None)

    final = ReshapeLayer(network,
                         shape=([0], -1))

    return final


# input_shape = (size, channel, width, height)
def build_st_network(input_shape):
    # General Params
    ini = lasagne.init.HeUniform()

    # SP Param
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()  # identity transform

    # Localization Network
    l_in = lasagne.layers.InputLayer(shape=(None, input_shape[1], input_shape[2], input_shape[3]))

    l_loc = lasagne.layers.MaxPool2DLayer(l_in,
                                          pool_size=(2, 2))

    l_loc = Conv2DLayer(l_loc,
                        num_filters=32,
                        filter_size=(5, 5),
                        stride=2,
                        W=ini)
    l_loc = MaxPool2DLayer(l_loc,
                           pool_size=(2, 2))

    l_loc = Conv2DLayer(l_loc,
                        num_filters=64,
                        filter_size=(5, 5),
                        stride=2,
                        W=ini)

    l_loc = MaxPool2DLayer(l_loc,
                           pool_size=(2, 2))

    l_loc = DenseLayer(l_loc,
                       num_units=64,
                       W=lasagne.init.HeUniform('relu'))

    l_loc = DenseLayer(l_loc,
                       num_units=6,
                       b=b,
                       W=lasagne.init.Constant(0.0),
                       nonlinearity=lasagne.nonlinearities.identity)

    # Transformer Network
    l_trans = TransformerLayer(l_in,
                               l_loc,
                               downsample_factor=1.0)

    l_trans = ScaleLayer(l_trans)

    final = ReshapeLayer(l_trans,
                         shape=([0], -1))
    return final
