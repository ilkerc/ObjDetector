import lasagne
import theano.tensor as T
import numpy as np
import theano
from lasagne.layers import Layer


class DiscreteLayer(Layer):

    def __init__(self, incoming, bins, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.bins = self.add_param(bins, shape=bins.shape, name='bins', trainable=False)

    def get_output_for(self, input, **kwargs):
        if not kwargs['withdiscrete']:
            return input
        else:
            theta = input
            theta = T.reshape(theta, (-1, 6))  # 6 because affine has 6 parameters
            dist = self.bins.transpose() - theta.flatten()
            arg_min = T.argmin(abs(dist), axis=0)
            arg_min_c = T.cast(arg_min, 'int32')
            return T.reshape(self.bins[0, arg_min_c], (-1, 6))
            return T.reshape(T.choose(arg_min, self.bins), (-1, 6))
"""
mins = T.min(abs(dist), axis=0)
#mins = T.reshape(mins, (-1, 6))

mins_sign = T.min(self.bins.transpose() - theta.flatten(), axis=0)
#mins_sign = T.reshape(mins_sign, (-1, 6))

diff = mins_sign - mins


arg_min = T.argmin(abs(dist), axis=0)
arg_min_c = T.cast(arg_min, 'int32')

return T.choose(arg_min_c, self.bins)
return T.reshape(arg_min , (-1 ,6))

theano.tensor.discrete_dtypes
x = mins-mins_sign

return theta - mins_sign




new_theta = self.bins[0, mins]
new_theta = new_theta.reshape(theta.shape)
new_theta = T.cast(new_theta, 'float32')
#up = new_theta - theta
#new_theta = discrete(theta, self.bins)

#return lasagne.nonlinearities.linear(new_theta)
"""

# Discrete assignment
def discrete(theta, bins):
    theta = T.reshape(theta, (-1, 6))  # 6 because affine has 6 parameters
    # bins_choose = _linspace(-3, 3, 100)
    # t_size = T.prod(theta.shape)
    # bins = T.tile(bins_choose, t_size).reshape((t_size, -1))
    dist = (bins.transpose() - theta.flatten())**2
    mins = T.argmin(dist, axis=0)
    new_theta = bins[0, mins]
    new_theta = new_theta.reshape(theta.shape)
    new_theta = T.cast(new_theta, 'float32')
    return new_theta


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