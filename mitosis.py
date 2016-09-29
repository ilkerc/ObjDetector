from utils.utils import iterate_minibatches, test_eval, show_network, test_histogram, \
    rescaler, plot_agumented_images, train_test_splitter
from Models import build_mitosis_encoder
from sampleFactory import generateSamples
import lasagne
import theano
import time
import theano.tensor as T
import numpy as np

# TODO : Regularisation is missing

# Params
num_epochs = 500
use_class = 1  # 1 is for mitosis
aug_sample = 50
batch_size = 1
encode_size = 50
use_st = False
regularize = False
reg_weight = 0.001

# Import The Data
samples_path_x = '/home/ilker/bap/mitosis/OriginalSampleX.npy'
samples_path_y = '/home/ilker/bap/mitosis/OriginalSampleY.npy'
X = np.reshape(np.load(samples_path_x), (-1, 50, 50, 3))
Y = np.load(samples_path_y)

# Augment Samples
X_set = [X[i] for i in range(0, X.shape[0]) if Y[i] == use_class]
Y_aug, X_aug = generateSamples(X_set, aug_sample)

# Construct the model
Xs = np.transpose(np.asarray(X_aug), (0, 3, 2, 1))
Ys = np.transpose(np.asarray(Y_aug), (0, 3, 2, 1))
network = build_mitosis_encoder(Xs.shape, encoding_size=encode_size, withst=use_st)

# params
params = lasagne.layers.get_all_params(network, trainable=True)
X = T.tensor4('inputs', dtype=theano.config.floatX)
Y = T.tensor4('targets', dtype=theano.config.floatX)
output = lasagne.layers.get_output(network, X, deterministic=False)
l_encoder = next(l for l in lasagne.layers.get_all_layers(network) if l.name is 'encoder')
encoder = lasagne.layers.get_output(l_encoder, X)

# Train Functions
cost = T.mean(lasagne.objectives.squared_error(output, Y))  # ssim(output, y) -> ssim is also a good metric
if regularize:
    all_layers = lasagne.layers.get_all_layers(network)
    l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * reg_weight
    cost += l2_penalty

updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=0.01, momentum=0.9)
train_func = theano.function([X, Y], [cost, output], updates=updates, allow_input_downcast=True)
encode_func = theano.function([X], [encoder], allow_input_downcast=True)
eval_func = theano.function([X], [output], allow_input_downcast=True)

# Training
print("Training Is About to Start")
trn_hist = np.zeros(num_epochs)
try:
    for epoch in range(num_epochs):
        # Training
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(Xs, Ys, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_func(inputs, targets)[0]
            train_batches += 1
            trn_hist[epoch] = train_err / train_batches

        # Print Results
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

except KeyboardInterrupt:
    print("Training is interrupted")
    pass
print("Completed, saved")
