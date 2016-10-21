import theano
import theano.tensor as T
from theano.gof import Op


class DummyOperation(Op):

    __props__ = ("name", "fn")

    def __init__(self, x, y):
        super(DummyOperation, self).__init__()
        self.x = x
        self.y = y
        T.flatten()
    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inp, out):
        x, y = inp
        z, = out
        z[0] = self.fn(x, y)

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]
