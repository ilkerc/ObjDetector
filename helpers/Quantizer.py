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
        self.rv_n = self.srng.normal((6, 1))
        super(Quantizer, self).__init__()

    def perform(self, node, inputs, output_storage):
        # Input & output storage settings
        x = inputs[0]
        y = inputs[1]
        out = output_storage[0]

        # Calculation
        new_theta = y * np.floor((x/y) + .5)

        # Add random noisy factor y/5 * random vector
        if self.addnoise:
            new_theta += (y * .2) * self.rv_n.eval()

        # Output Setting
        out[0] = new_theta

    # TODO: Investigate Output Gradients,
    # TODO: If we decide to include ranges as learning parameters, hereby we need to define their gradients
    def grad(self, inputs, output_grads):
        return [output_grads[0], output_grads[0]]

if __name__ == "__main__":
    theano.config.exception_verbosity = 'high'
    op = Quantizer()

    x = theano.tensor.fmatrix()
    y = theano.tensor.fvector()
    f = theano.function([x, y], [op(x, y), T.grad(T.constant(1.1), x)], mode='DebugMode', allow_input_downcast=True)

    y1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    x1 = np.random.rand(3, 6)

    print op(x1, y1)
