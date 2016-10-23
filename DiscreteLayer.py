import theano
import theano.tensor as T
from lasagne.layers import Layer
from lasagne.init import Constant
from helpers.DiscOP import DiscOP


class DiscreteLayer(Layer):

    def __init__(self, incoming, start=Constant(-3.), stop=Constant(3.), linrange=Constant(50.), **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.start = self.add_param(start, (1,), name='start', trainable=False)
        self.stop = self.add_param(stop, (1,), name='stop', trainable=False)
        self.linrange = self.add_param(linrange, (1,), name='linrange', trainable=False)
        self.op = DiscOP()

    def get_output_for(self, inputs, **kwargs):
        theta = inputs
        bins = self.make_bins(inputs.shape[0])

        return self.op(bins, theta)

    def make_bins(self, b_size):
        bins_choose = DiscreteLayer._linspace(self.start, self.stop, self.linrange)
        t_size = b_size * 6
        bins = T.tile(bins_choose, t_size).reshape((t_size, -1))
        return T.cast(bins, theano.config.floatX)

    @staticmethod
    def _linspace(start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = T.cast(start[0], theano.config.floatX)
        stop = T.cast(stop[0], theano.config.floatX)
        num = T.cast(T.floor(num[0]), theano.config.floatX)
        step = (stop-start)/(num-1)
        return T.arange(num, dtype=theano.config.floatX)*step+start
