import theano
import theano.tensor as T
import numpy as np


class DiscOP(theano.Op):
    """
    This creates an Op that takes x to a*x+b.
    """
    __props__ = ()

    itypes = [theano.tensor.dmatrix, theano.tensor.dmatrix]
    otypes = [theano.tensor.dmatrix]

    def perform(self, node, inputs, output_storage):
        # Input & output storage settings
        x = inputs[1]
        y = inputs[0]
        out = output_storage[0]

        # Calculation
        dist = y.transpose() - x.flatten()
        arg_min = np.argmin(abs(dist), axis=0)
        new_theta_flat = y[0, arg_min]
        new_theta = np.reshape(new_theta_flat, x.shape)

        # Output Setting
        out[0] = new_theta

    # TODO: Investigate Output Gradients
    def grad(self, inputs, output_grads):
        return [output_grads[0], output_grads[0]]


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start

if __name__ == "__main__":
    theano.config.exception_verbosity = 'high'
    b_size = 2
    bins_choose = np.linspace(-1, 1, 5)
    t_size = b_size*6
    bins = np.tile(bins_choose, t_size).reshape((t_size, -1))

    disc_operator = DiscOP()

    x = theano.tensor.fmatrix()
    y = theano.tensor.fmatrix()
    f = theano.function([x, y], disc_operator(x, y), mode='DebugMode', allow_input_downcast=True)

    x1 = bins
    y1 = np.random.rand(b_size, 6)

    print y1
    print "\n\n\n"
    print f(x1, y1)
