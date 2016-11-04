import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams


class Quantizer(theano.Op):
    """
    This creates an Op that takes x to a*x+b.
    """
    #__props__ = ("quants", )

    itypes = [theano.tensor.fmatrix, theano.tensor.fvector]
    otypes = [theano.tensor.fmatrix]

    def __init__(self, addnoise=True):
        self.srng = RandomStreams()
        self.addnoise = addnoise
        self.rv_n = self.srng.uniform((6, ))
        super(Quantizer, self).__init__()

    def perform(self, node, inputs, output_storage):
        # Input & output storage settings
        x = inputs[0]
        y = inputs[1]
        out = output_storage[0]

        # Calculation
        if self.addnoise:
            x_noise = x + ((x * .05) * (self.rv_n.eval() - .5))
            new_theta = y * np.floor((x_noise/y) + .5)
        else:
            new_theta = y * np.floor((x/y) + .5)

        out[0] = new_theta

    # TODO: Investigate Output Gradients,
    # TODO: If we decide to include ranges as learning parameters, hereby we need to define their gradients
    def grad(self, inputs, output_grads):
        theta, quant = inputs
        # Quantizers gradients are zero ?
        return [output_grads[0], T.ones_like(quant)]

if __name__ == "__main__":
    theano.config.exception_verbosity = 'high'
    op = Quantizer()

    x = theano.tensor.fmatrix()
    y = theano.tensor.fvector()
    f = theano.function([x, y], [op(x, y)], mode='DebugMode', allow_input_downcast=True)

    y1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype='float32')
    x1 = np.random.rand(3, 6).astype('float32')

    print f(x1, y1)
