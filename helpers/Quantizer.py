import theano
import theano.tensor as T
import numpy as np


class Quantizer(theano.Op):
    """
    This creates an Op that takes x to a*x+b.
    """
    __props__ = ("mins", "maxs", "ranges")

    itypes = [theano.tensor.fmatrix]
    otypes = [theano.tensor.fmatrix]

    def __init__(self, quants):
        self.quants = quants
        super(Quantizer, self).__init__()

    def perform(self, node, inputs, output_storage):
        # Input & output storage settings
        x = inputs[0]
        out = output_storage[0]

        # Calculation
        new_theta = self.quant * T.floor((x/self.quant) + .5)

        # Output Setting
        out[0] = new_theta

    # TODO: Investigate Output Gradients,
    # TODO: If we decide to include ranges as learning parameters, hereby we need to define their gradients
    def grad(self, inputs, output_grads):
        return [output_grads[0]]
