def newQuantizer(self, inputs, quantizers):
    shape = inputs.shape
    outputs = np.zeros(shape=shape, dtype=theano.config.floatX)

    # Theta Iterator
    for i in range(shape[1]):
        theta_i = inputs[:, i]

        # Batch Iterator
        for j in range(shape[0]):
            theta = theta_i[j]

            # Theta Optimizer
            q = quantizers[i]
            x_i = theta
            x_o = self.Qnt(x_i, q)

            # Optimizer
            while(np.abs(x_o - x_i) > self.eps):
                q = q / 2.0
                x_o = self.Qnt(x_i, q)

            # End of Optimisation
            outputs[j, i] = x_o

    return outputs


def Qnt(self, x, y):
    return y * np.floor((x/y) + .5)






def quantization_devider(dist):
    # Important: dist should be from the epoch at t-1
    # Get the parameters from the network
    quant = params[8] # the quantization parameter as shared
    quant_val = quant.get_value()
    quant_units = quant_val.shape[0]
    changes = []

    # Repeat for each Unit
    for i in range(quant_units):

        # Every quantization number is insterested with one distribution (unit outputs in an epoch)
        T_i = dist[:, i]
        Q = quant_val[i]

        # [Q at t-1] Calculate min, max, range values to obtain the histogram
        T_i_min = np.min(T_i)
        T_i_max = np.max(T_i)
        T_i_bins = np.arange(T_i_min, T_i_max, Q)

        # [Q at t-1] Calculate histogram of
        h_d, _ = np.histogram(T_i, bins=T_i_bins)
        h_d = h_d / float(h_d.sum()) # Convert the histogram into p.density
        h_d_std = np.std(h_d)
        h_d_max = np.max(h_d)
        h_d_nonzero = np.count_nonzero(h_d)

        # [Q at t] Calculate min, max, range values to obtain the histogram, mins and maxes are the same
        T_i_bins_t = np.arange(T_i_min, T_i_max, Q*.7)

        # [Q at t] Calculate histogram of
        h_d_t, _ = np.histogram(T_i, bins=T_i_bins_t) # Approximated bins of t over time t-1
        h_d_t = h_d_t / float(h_d_t.sum()) # Convert the histogram into p.density
        h_d_t_max = np.max(h_d_t)
        h_d_std_t = np.std(h_d_t)
        h_d_nonzero_t = np.count_nonzero(h_d_t)

        # Because we want a small std, and make the distribution look uniform
        if  h_d_t_max < h_d_max: # np.prod(h_d_t) > 0:
            Q = Q * .7
            changes.append("idx: {0}, val: {1:.5}".format(i, Q))

        # Change Q
        quant_val[i] = Q

    # Report
    if len(changes) > 1:
        print "Changes occured in units: " + ", ".join([str(x) for x in changes] )

    # At the end, update the parameters
    quant.set_value(np.array(quant_val, dtype=theano.config.floatX))
