import lasagne


class DiscreteLayer(lasagne.layers.MergeLayer):

    def __init__(self, incoming, **kwargs):
        super(DiscreteLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shapes):
        shp = input_shapes[0]
        return list(shp[:2]) + [int(s) for s in shp[2:]]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        print inputs
        return inputs