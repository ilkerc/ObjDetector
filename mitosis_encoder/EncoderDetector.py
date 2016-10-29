from Models import build_mitosis_encoder
import lasagne
import theano
import theano.tensor as T
import numpy as np

"""
Weight Path -> Pre-Trained weights path
Image Path -> (None, 3, 50, 50)
Encode_Size -> Encoder num, THIS HAS TO BE SAME AS TRAINING
Use_st -> When set true ST

Example Usage :

a = EncoderDetector('model_ae_26.npz', (None, 3, 50, 50), 50, False)
xx = a.get_encoded_feas(list(Xs[0:500]))

"""


class EncoderDetector:

    def __init__(self, weight_path, image_shape, encode_size, use_st):
        self.weight_path = weight_path

        # Construct the model
        network = build_mitosis_encoder(image_shape, encoding_size=encode_size, withst=use_st)

        # params
        X = T.tensor4('inputs', dtype=theano.config.floatX)
        output = lasagne.layers.get_output(network, X, deterministic=False)
        l_encoder = next(l for l in lasagne.layers.get_all_layers(network) if l.name is 'encoder')
        encoder = lasagne.layers.get_output(l_encoder, X)

        # Functions
        self.encode_func = theano.function([X], [encoder], allow_input_downcast=True)
        self.eval_func = theano.function([X], [output], allow_input_downcast=True)

        # Restore Weights
        with np.load(weight_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    "Input Image has to be list of (3, 50, 50) or whatever the shape was during training"
    def get_encoded_feas(self, images_list):
        outputs = []
        for img in images_list:
            encoded_feas = self.encode_func(np.expand_dims(img, axis=0))
            encoded_feas = np.asarray(encoded_feas).flatten()
            outputs.append(encoded_feas)
        return np.asarray(outputs)
