from lasagne.layers import Layer
from helpers.DiscOP import DiscOP


class DiscreteLayer(Layer):

    def __init__(self, incoming, mins, maxs, ranges, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.op = DiscOP(mins=mins, maxs=maxs, ranges=ranges)

    def get_output_for(self, inputs, **kwargs):
        theta = inputs
        return self.op(theta)