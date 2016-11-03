from lasagne.layers import Layer
from helpers.DiscOP import DiscOP
import theano.tensor as T


class DiscreteLayer(Layer):

    def __init__(self, incoming, mins, maxs, ranges, quant, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.quant = self.add_param(quant, quant.shape, name='quant', trainable=False)
        self.op = DiscOP(mins=mins, maxs=maxs, ranges=ranges)

    def get_output_for(self, inputs, **kwargs):
        theta = inputs
	return self.quant * T.floor((theta/self.quant) + .5)
        #return 0.12 * T.floor((theta / 0.12) + (1 / 2))
        #return self.op(theta)
