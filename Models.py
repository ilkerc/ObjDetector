import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import ReshapeLayer, DenseLayer, InputLayer, TransformerLayer, ScaleLayer, Upscale2DLayer, TransposedConv2DLayer

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer

    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayer
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer

    print('Using lasagne.layers (slower)')


# This builds a model of Conv. Autoencoder
def build_cnnae_network_2(input_shape):

    conv_filters = 16
    filter_size = 5
    pool_size = 2
    encode_size = 512

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=(pool_size, pool_size))

    l_reshape1 = ReshapeLayer(l_pool1, shape=([0], -1))

    l_encode = DenseLayer(l_reshape1,
                          name='encode',
                          num_units=encode_size)

    l_decode = DenseLayer(l_encode,
                          W=l_encode.W.T,
                          num_units=l_reshape1.output_shape[1])

    l_reshape2 = ReshapeLayer(l_decode,
                              shape=([0],
                                     conv_filters, 30, 30))

    l_unpool1 = Upscale2DLayer(l_reshape2,
                               scale_factor=pool_size)

    l_de = TransposedConv2DLayer(l_unpool1,
                                 num_filters=l_conv1.input_shape[1],
                                 W=l_conv1.W,
                                 filter_size=l_conv1.filter_size,
                                 stride=l_conv1.stride,
                                 crop=l_conv1.pad,
                                 flip_filters=not l_conv1.flip_filters)

    l_output = ReshapeLayer(l_de,
                            shape=([0], -1))

    return l_output


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


# This builds a model of Conv. Autoencoder
def build_cnnae_network(input_shape):
    conv_filters = 32
    filter_size = 5
    pool_size = 2
    encode_size = input_shape[2] * 2

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=conv_filters,
                          filter_size=(filter_size, filter_size),
                          nonlinearity=None)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=(pool_size, pool_size))

    l_reshape1 = ReshapeLayer(l_pool1,
                              shape=([0], -1))

    l_encode = DenseLayer(l_reshape1,
                          name='encode',
                          num_units=encode_size)

    l_decode = DenseLayer(l_encode,
                          num_units=l_reshape1.output_shape[
                              1])  # num_units=conv_filters * (input_shape[2] + filter_size - 1) ** 2 / 4)

    l_reshape2 = ReshapeLayer(l_decode,
                              shape=([0],
                                     conv_filters,
                                     (input_shape[2] - pool_size - 1) / 2,
                                     (input_shape[2] - pool_size - 1) / 2))

    l_unpool1 = Upscale2DLayer(l_reshape2,
                               scale_factor=pool_size)

    l_deconv1 = Conv2DLayer(l_unpool1,
                            num_filters=input_shape[1],
                            filter_size=(filter_size, filter_size),
                            pad='full',
                            nonlinearity=None)

    l_output = ReshapeLayer(l_deconv1,
                            shape=([0], -1))

    return l_output
