import lasagne
import theano.tensor as T
import numpy as np
import ipdb
import theano


class DiscreteLayer(lasagne.layers.Layer):

    def __init__(self, incoming, bin_count, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.bin_count = bin_count

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        theta = inputs
        new_theta = dis2(theta, self.bin_count)
        return new_theta


def discrete(theta, bins):
    """
    import numpy
    data = 5 * numpy.random.random(100) - 5
    bins = numpy.linspace(-5, 0, 10)
    digitized = numpy.digitize(data, bins)
    bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
    :param theta:
    :param bins:
    :return:
    """
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


def dis3(theta):
    theta = T.reshape(theta, (-1, 6))
    bins = _linspace(-3, 3, 100)

    t_1 = T.argmin(theta[0, :], )

def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start