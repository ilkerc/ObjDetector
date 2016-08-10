import numpy as np
import theano
import lasagne
import theano.tensor as T
from lasagne import layers
from lasagne.layers import ReshapeLayer, DenseLayer, InputLayer,\
    TransformerLayer, ScaleLayer, Upscale2DLayer, TransposedConv2DLayer,\
    DropoutLayer, TPSTransformerLayer

try:
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    print('Using cuda_convnet (faster)')
except ImportError:
    from lasagne.layers import Conv2DLayer as Conv2DLayer
    from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
    print('Using lasagne.layers (slower)')


# Spatial Transformer Network with spline
def build_st_spline_network(input_shape):
    W = b = lasagne.init.Constant(0.0)
    num_points = 16
    num_filters = 64
    filter_size = (3, 3)
    pool_size = (2, 2)

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=pool_size)

    l_conv2 = Conv2DLayer(l_pool1,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool2 = MaxPool2DLayer(l_conv2,
                             pool_size=pool_size)

    l_dense1 = DenseLayer(l_pool2,
                          num_units=128)

    l_dense2 = DenseLayer(l_dense1,
                          num_units=num_points*2,
                          W=W,
                          b=b,
                          nonlinearity=None)

    l_st = TPSTransformerLayer(l_in,
                               l_dense2,
                               control_points=num_points)

    l_output = ReshapeLayer(l_st,
                            shape=([0], -1))

    return l_output


# This builds a model of Conv. Autoencoder
def build_cnnae_network(input_shape):

    conv_filters = 16
    filter_size = 5
    pool_size = 2
    encode_size = input_shape[2] * 2

    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_dropout1 = DropoutLayer(l_in,
                              p=0.5)

    l_conv1 = Conv2DLayer(l_dropout1,
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
                                     conv_filters,
                                     int(np.sqrt(l_reshape1.output_shape[1] / conv_filters)),
                                     int(np.sqrt(l_reshape1.output_shape[1] / conv_filters))))

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
    num_filters = 64
    filter_size = (3, 3)
    pool_size = (2, 2)

    # SP Param
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()  # identity transform

    # Localization Network
    l_in = InputLayer(shape=(None,
                             input_shape[1],
                             input_shape[2],
                             input_shape[3]))

    l_conv1 = Conv2DLayer(l_in,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool1 = MaxPool2DLayer(l_conv1,
                             pool_size=pool_size)

    l_conv2 = Conv2DLayer(l_pool1,
                          num_filters=num_filters,
                          filter_size=filter_size)

    l_pool2 = MaxPool2DLayer(l_conv2,
                             pool_size=pool_size)

    l_loc = DenseLayer(l_pool2,
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
def build_cnnae_network_deprecated(input_shape):
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
