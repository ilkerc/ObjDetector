from lasagne.layers import Layer
from helpers.DiscOP import DiscOP
from helpers.Quantizer import Quantizer
import theano.tensor as T


class DiscreteLayer(Layer):

    def __init__(self, incoming, mins, maxs, ranges, quant, addNoise=True, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)
        self.quant = self.add_param(quant, quant.shape, name='quant', trainable=False)
        #self.op = DiscOP(mins=mins, maxs=maxs, ranges=ranges)
        self.op = Quantizer(addnoise=addNoise, eps=0.001)

    def get_output_for(self, inputs, **kwargs):
        theta = inputs
        return self.op(theta, T.as_tensor_variable(self.quant))
        #return self.quant * T.floor((theta/self.quant) + .5)
        #return 0.12 * T.floor((theta / 0.12) + (1 / 2))
        #return self.op(theta)


if __name__ == "__main__":
    import theano
    import numpy as np
    t = theano.tensor.fmatrix()
    q = theano.tensor.fvector()

    theta = np.array([[1.121, 2, 3, 4, 5, 6], [1.04, 2, 3, 4, 5, 6]])
    quant = np.array([0.008, 0.004, 0.019, 0.007, 0.006, 0.023], dtype='float32')

    f = theano.function([t, q], q * T.floor((t/q) + .5), mode='DebugMode', allow_input_downcast=True)

    print f(theta, quant)
