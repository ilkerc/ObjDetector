from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import lasagne
import theano
import theano.tensor as T
import numpy as np
from utils.Augmentor import Augmentor
from utils.utils import iterate_minibatches, test_eval, show_network, set_divider, rescaler, compute_reconstruction_error
from Models import build_st_network, build_cnnae_network, build_st_spline_network
from scipy import misc

# Constants
BATCH_SIZE = 30
WINDOW_SIZE = 64
# Global Variables
agumented_samples = None
original_samples = None


def train_network(x_data, y_data, num_epochs=100, batch_size=BATCH_SIZE, model='cnn', shift_target=False):
    # Make Y a matrix
    X_tr, Y_tr, X_val, Y_val, X_tst, Y_tst = set_divider(x_data, y_data)

    # Randomly select 10 percent of inputs and assign it to targets, this may reduce overfitting
    if shift_target:
        r_tr = np.random.randint(0, Y_tr.shape[0], int(Y_tr.shape[0]*.5), dtype='int32')
        r_tst = np.random.randint(0, Y_tst.shape[0], int(Y_tst.shape[0]*.5), dtype='int32')
        r_val = np.random.randint(0, Y_val.shape[0], int(Y_val.shape[0]*.5), dtype='int32')
        Y_tr[r_tr] = X_tr[r_tr]
        Y_val[r_val] = X_val[r_val]
        Y_tst[r_tst] = X_tst[r_tst]
        # Y_tr = np.random.permutation(np.reshape(X_tr, (X_tr.shape[0], -1)))
        # Y_tst = np.random.permutation(np.reshape(X_tst, (X_tst.shape[0], -1)))
        # Y_val = np.random.permutation(np.reshape(X_val, (X_val.shape[0], -1)))

    # Reshape to vector
    Y_val = np.reshape(Y_val, (Y_val.shape[0], -1))
    Y_tst = np.reshape(Y_tst, (Y_tst.shape[0], -1))
    Y_tr = np.reshape(Y_tr, (Y_tr.shape[0], -1))

    # Building the model
    if model == 'cnn':
        network = build_cnnae_network(X_tr.shape)
    elif model == 'st':
        network = build_st_network(X_tr.shape)
    elif model == 'st_sp':
        network = build_st_spline_network(X_tr.shape)
    else:
        print("No such Model")
        return

    params = lasagne.layers.get_all_params(network, trainable=True)
    print("Model is ready for training")

    # Trainer Function and Variable Holders
    X = T.tensor4('inputs', dtype=theano.config.floatX)
    Y = T.matrix('targets', dtype=theano.config.floatX)

    # Train Functions
    output = lasagne.layers.get_output(network, X, deterministic=False)
    cost = T.mean(lasagne.objectives.squared_error(output, Y))
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=0.01, momentum=0.9)
    train_func = theano.function([X, Y], [cost, output], updates=updates, allow_input_downcast=True)
    eval_func = theano.function([X], [output], allow_input_downcast=True)

    # Test Functions
    test_prediction = lasagne.layers.get_output(network, X, deterministic=True)
    test_loss = T.mean(lasagne.objectives.squared_error(test_prediction, Y))
    test_func = theano.function([X, Y], [test_loss], allow_input_downcast=True)

    # Reconstruction error Function
    reconstruction_prediction = lasagne.layers.get_output(network, X, deterministic=True)
    mu_loss = compute_reconstruction_error(X, Y, reconstruction_prediction)
    reconstruction_func = theano.function([X, Y],
                                          [mu_loss, reconstruction_prediction],
                                          allow_input_downcast=True)

    # Training, Validating Iterator
    print("Training Is About to Start")
    try:
        for epoch in range(num_epochs):
            # Training
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_tr, Y_tr, batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_func(inputs, targets)[0]
                train_batches += 1

            # Validation
            val_err = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, Y_val, batch_size, shuffle=False):
                inputs, targets = batch
                err = test_func(inputs, targets)
                val_err += err[0]
                val_batches += 1

            # Print Results
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    except KeyboardInterrupt:
        print("Training is interrupted")
        pass
    print("Completed, saved")

    # Calculate reconstruction error
    inputs = X_tst
    targets = Y_tst
    loss, reconstruction_prediction = reconstruction_func(inputs, targets)
    print("Reconstruction error : {:0}".format(float(loss)))

    # # After training, we compute and print the test error:
    # test_err = 0
    # test_batches = 0
    # for batch in iterate_minibatches(X_tst, Y_tst, batch_size, shuffle=False):
    #     inputs, targets = batch
    #     err = test_func(inputs, targets)
    #     test_err += err[0]
    #     test_batches += 1
    # print("Final results:")
    # print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    return eval_func, train_func, network


# Callback from plot click
def selected_callback(os, ass):
    global agumented_samples
    global original_samples
    original_samples = os
    agumented_samples = ass
    print("Agumented Samples Has Been Created")


# Main Run
#theano.config.exception_verbosity='high'
#theano.config.optimizer='None'
lasagne.random.set_rng(np.random.RandomState(1234))  # Set random state so we can investigate results
path = '../me.png'
coords = {'leye': (420, 402),
          'reye': (443, 575),
          'hpho': (879, 360),
          'e1'  : (700, 100),
          'e2'  : (500, 600),
          'e3'  : (652, 820)}
img = misc.imread(path, flatten=False)  # Set flatten true if working w/ gray i mages
img = img[:, :, :3]  # Just in case if the image has more then 3 channels like alpha
aug_count = 100
rescale_factor = 0.25
a = Augmentor(selected_callback, img_data=img, window_size=WINDOW_SIZE, count=aug_count)
Ys, Xs = a.manuel(coords['leye'])
Ys, Xs = rescaler(Xs, Ys, rescale_factor=rescale_factor)

evl_func, trn_func, ntwrk = train_network(Xs,
                                          Ys,
                                          num_epochs=100,
                                          batch_size=1,
                                          model='st',
                                          shift_target=False)

X_tst = set_divider(Xs, Ys)[4]  # Get the test data for a hones test :)
test_eval(evl_func, X_tst, shape=(-1, img.shape[2],
                                  WINDOW_SIZE*rescale_factor,
                                  WINDOW_SIZE*rescale_factor), show_all=True)

# Visualize
# show_network(ntwrk)
