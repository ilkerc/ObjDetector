from helpers.utils import iterate_minibatches
from Models import build_mitosis_encoder
from sampleFactory import generateSamples
import lasagne
import theano
import time
import theano.tensor as T
import numpy as np

# Params
num_epochs = 500
use_class = 1  # 1 is for mitosis
aug_sample = 5
batch_size = 5
encode_size = 100
use_st = True
regularize = True
reg_weight = 0.0005

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

updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=0.05, momentum=0.9)
train_func = theano.function([X, Y], [cost, output], updates=updates, allow_input_downcast=True)
encode_func = theano.function([X], [encoder], allow_input_downcast=True)
eval_func = theano.function([X], [output], allow_input_downcast=True)

# Training
print("Training Is About to Start")
trn_hist = np.zeros(num_epochs)
try:
    for epoch in range(num_epochs):
        # Save the model
        if epoch % 20 == 0:
            np.savez('model_' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(network))

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
    np.savez('model_' + str(epoch) + '.npz', *lasagne.layers.get_all_param_values(network))
    pass
print("Completed, saved")


"""
Testing After Here
Assuming that the code blocks that builds the model has already been executed


# Load Test Data

# Restore params
with np.load('model_26.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)

# Test 1 -> Encoded Features
one_img = Xs[0]
encoded_feas = encode_func(np.expand_dims(one_img, axis=0))
encoded_feas = np.asarray(encoded_feas).flatten()

# Test 2 -> Get output
one_img = Xs[0]
out_out = eval_func(np.expand_dims(one_img, axis=0))
out_out = np.asarray(out_out).squeeze().transpose()

"""
