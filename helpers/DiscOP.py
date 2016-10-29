import theano
import theano.tensor as T
import numpy as np


class DiscOP(theano.Op):
    """
    This creates an Op that takes x to a*x+b.
    """
    __props__ = ("mins", "maxs", "ranges")

    itypes = [theano.tensor.fmatrix]
    otypes = [theano.tensor.fmatrix]

    def __init__(self, mins, maxs, ranges):
        self.mins = mins
        self.maxs = maxs
        self.ranges = ranges
        super(DiscOP, self).__init__()

    def perform(self, node, inputs, output_storage):
        # Input & output storage settings
        x = inputs[0]
        out = output_storage[0]

        # Calculation
        new_theta = self.discrete_theta(x)

        # Output Setting
        out[0] = new_theta

    # TODO: Investigate Output Gradients,
    # TODO: If we decide to include ranges as learning parameters, hereby we need to define their gradients
    def grad(self, inputs, output_grads):
        return [output_grads[0]]

    def discrete_theta(self, theta):
        theta = theta.reshape((-1, 6))
        batch_size = theta.shape[0]
        t_1 = np.tile(np.linspace(self.mins[0], self.maxs[0], self.ranges[0]), batch_size).reshape((batch_size, -1))
        t_2 = np.tile(np.linspace(self.mins[1], self.maxs[1], self.ranges[1]), batch_size).reshape((batch_size, -1))
        t_3 = np.tile(np.linspace(self.mins[2], self.maxs[2], self.ranges[2]), batch_size).reshape((batch_size, -1))
        t_4 = np.tile(np.linspace(self.mins[3], self.maxs[3], self.ranges[3]), batch_size).reshape((batch_size, -1))
        t_5 = np.tile(np.linspace(self.mins[4], self.maxs[4], self.ranges[4]), batch_size).reshape((batch_size, -1))
        t_6 = np.tile(np.linspace(self.mins[5], self.maxs[5], self.ranges[5]), batch_size).reshape((batch_size, -1))

        t_1_o = np.expand_dims(theta[:, 0], axis=0)
        t_2_o = np.expand_dims(theta[:, 1], axis=0)
        t_3_o = np.expand_dims(theta[:, 2], axis=0)
        t_4_o = np.expand_dims(theta[:, 3], axis=0)
        t_5_o = np.expand_dims(theta[:, 4], axis=0)
        t_6_o = np.expand_dims(theta[:, 5], axis=0)

        dist_t1 = abs(t_1_o.T - t_1)
        dist_t2 = abs(t_2_o.T - t_2)
        dist_t3 = abs(t_3_o.T - t_3)
        dist_t4 = abs(t_4_o.T - t_4)
        dist_t5 = abs(t_5_o.T - t_5)
        dist_t6 = abs(t_6_o.T - t_6)

        arg_min_t1 = np.argmin(abs(dist_t1), axis=1)
        arg_min_t2 = np.argmin(abs(dist_t2), axis=1)
        arg_min_t3 = np.argmin(abs(dist_t3), axis=1)
        arg_min_t4 = np.argmin(abs(dist_t4), axis=1)
        arg_min_t5 = np.argmin(abs(dist_t5), axis=1)
        arg_min_t6 = np.argmin(abs(dist_t6), axis=1)

        new_t1 = t_1[0, arg_min_t1]
        new_t2 = t_2[0, arg_min_t2]
        new_t3 = t_3[0, arg_min_t3]
        new_t4 = t_4[0, arg_min_t4]
        new_t5 = t_5[0, arg_min_t5]
        new_t6 = t_6[0, arg_min_t6]

        new_theta = np.squeeze(np.dstack([new_t1, new_t2, new_t3, new_t4, new_t5, new_t6]))

        return new_theta.astype('float32')

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
