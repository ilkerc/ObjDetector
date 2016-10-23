import theano
import theano.tensor as T
from lasagne.layers import Layer
from lasagne.init import Constant
from helpers.DiscOP import DiscOP


class DiscreteLayer(Layer):

    def __init__(self, incoming, inits, trainables, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.start = self.add_param(Constant(inits[0]), (1,), name='start', trainable=trainables[0])
        self.stop = self.add_param(Constant(inits[1]), (1,), name='stop', trainable=trainables[0])
        self.linrange = self.add_param(Constant(inits[2]), (1,), name='linrange', trainable=trainables[0])
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
