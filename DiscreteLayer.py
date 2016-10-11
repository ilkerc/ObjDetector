import lasagne
import theano.tensor as T
import numpy as np
import ipdb
import theano
from lasagne.layers import Layer


class DiscreteLayer(Layer):

    def __init__(self, incoming, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        theta = input
        new_theta = discrete(theta)

        return new_theta


# Discrete assignment
def discrete(theta):
    theta = T.reshape(theta, (-1, 6))  # 6 because affine has 6 parameters
    bins_choose = _linspace(-3, 3, 100)
    t_size = T.prod(theta.shape)
    bins = T.tile(bins_choose, t_size).reshape((t_size, -1))
    dist = (bins.transpose() - theta.flatten())**2
    mins = T.argmin(dist, axis=0)
    new_theta = bins_choose[mins]
    new_theta = new_theta.reshape(theta.shape)
    new_theta = T.cast(new_theta, 'float32')
    return lasagne.nonlinearities.linear(new_theta)


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


"""
def test(theta):
    return theta



    import numpy
    data = 5 * numpy.random.random(100) - 5
    bins = numpy.linspace(-5, 0, 10)
    digitized = numpy.digitize(data, bins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    :param theta:
    :param bins:
    :return:

    # For each transformation parameter create space between -3 & 3
    dis_vals = np.array([np.linspace(-3, 3, bins[i]) for i in range(len(bins))])

    # theta size is batch, params
    theta_np = lasagne.utils.floatX(theta)
    num_batch, num_params = theta_np.shape

    new_theta = np.empty_like(theta_np)
    for i in range(0, num_batch):
        data_i = theta_np[:, i]
        bins_i = dis_vals[i]
        digitized = np.digitize(data_i, bins_i)
        dis_i = [bins_i[digitized[i]] for i in range(len(data_i))]
        new_theta[i, :] = dis_i

    return T.as_tensor_variable(new_theta)


def dis2(theta, bins):
    dis_vals = np.array([np.linspace(-3, 3, bins[i]) for i in range(len(bins))])
    outputs, updates = theano.scan(fn=cc,
                                   sequences=[theta, dis_vals],
                                   n_steps=theta.shape[0])
    return outputs


def cc(theta, bins):
    #digitized = np.digitize(theta, bins)
    #dis_i = np.array([bins[digitized[i]] for i in range(len(bins))])
    return theta
"""