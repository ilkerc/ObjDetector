from __future__ import print_function
from nolearn.lasagne.visualize import plot_conv_weights
from utils.Augmentor import Augmentor
from utils.theano_utils import ssim, compute_reconstruction_error
from utils.utils import iterate_minibatches, test_eval, show_network, test_histogram, \
    rescaler, plot_agumented_images, train_test_splitter
from Models import build_st_network, build_cnnae_network, build_st_spline_network, build_cnnae_network_2conv
from scipy import misc
import matplotlib.pyplot as plt
import time
import lasagne
import theano
import theano.tensor as T
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

# Constants
BATCH_SIZE = 30
WINDOW_SIZE = 64
# Global Variables
agumented_samples = None
original_samples = None


def train_network(x_data, y_data, num_epochs=100, batch_size=BATCH_SIZE, model='cnn', shift_target=False):
    # Make Y a matrix
    Xtr, Xtst, Ytr, Ytst = train_test_splitter(x_data, y_data, ratio=0.2, seed=42)

    # Randomly select x percent of inputs and assign it to targets, this may reduce overfitting
    if shift_target:
        r_selected = np.random.choice(Xtr.shape[0], int(Xtr.shape[0]) * .5, replace=False)
        Ytr[r_selected] = Xtr[r_selected]

    # Reshape to vector
    Ytst = np.reshape(Ytst, (Ytst.shape[0], -1))
    Ytr = np.reshape(Ytr, (Ytr.shape[0], -1))

    # Building the model
    if model == 'cnn':
        network = build_cnnae_network(Xtr.shape)
    elif model == 'st':
        network = build_st_network(Xtr.shape)
    elif model == 'st_sp':
        network = build_st_spline_network(Xtr.shape)
    elif model == 'cnn2':
        network = build_cnnae_network_2conv(Xtr.shape)
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
    cost = T.mean(lasagne.objectives.squared_error(output, Y)) + ssim(output, Y)
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
    trn_hist = np.zeros(num_epochs)
    try:
        for epoch in range(num_epochs):
            # Training
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(Xtr, Ytr, batch_size, shuffle=True):
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

    # Calculate reconstruction error
    inputs = Xtst
    targets = Ytst
    loss, reconstruction_prediction = reconstruction_func(inputs, targets)
    print("Reconstruction error : {:0}".format(float(loss)))

    # Calculate test history
    test_hist = test_histogram(Xtst, Ytst, test_func)

    return eval_func, train_func, test_func, reconstruction_func, network, test_hist


# Callback from plot click
def selected_callback(os, ass):
    global agumented_samples
    global original_samples
    original_samples = os
    agumented_samples = ass
    print("Agumented Samples Has Been Created")


# Main Run
# theano.config.exception_verbosity='high'
# theano.config.optimizer='None'
lasagne.random.set_rng(np.random.RandomState(1234))  # Set random state so we can investigate results
path = '../me.png'
coords = {'leye': (420, 402),
          'reye': (443, 575),
          'hpho': (879, 360),
          'e1': (700, 100),
          'e2': (500, 600),
          'e3': (652, 820)}
img = misc.imread(path, flatten=False)  # Set flatten true if working w/ gray i mages
img = img[:, :, :3]  # Just in case if the image has more then 3 channels like alpha
rescale_factor = 0.25
a = Augmentor(selected_callback,
              img_data=img,
              window_size=WINDOW_SIZE,
              scale_to_percent=1.2,
              rotation_deg=(0, 360),
              shear_deg=(-20, 20),
              translation_x_px=5,
              translation_y_px=5,
              transform_channels_equally=True
              )
root_key = 'leye'
Ys, Xs = a.manuel(coords[root_key])
Ys, Xs = rescaler(Xs, Ys, rescale_factor=rescale_factor)
X_tr, X_tst, Y_tr, Y_tst = train_test_splitter(Xs, Ys, ratio=0.2, seed=42)
evl_func, trn_func, tst_func, recon_func, ntwrk, train_hist = train_network(Xs,
                                                                            Ys,
                                                                            num_epochs=400,
                                                                            batch_size=5,
                                                                            model='cnn2',
                                                                            shift_target=False)

# Plotter For test
for key, val in coords.items():
    if key == root_key:
        continue
    Ys, Xs = a.manuel(val)
    Ys, Xs = rescaler(Xs, Ys, rescale_factor=rescale_factor)
    X_tst = train_test_splitter(Xs, Ys, ratio=0.2, seed=42)[1]
    tst_hist = test_histogram(X_tst, Y_tst, tst_func)  # Y_tst -> Train targets

    plt.figure()
    plt.title(key + ' vs ' + root_key)
    plt.plot(tst_hist, label='Eval (Unseen Samples) Error', linewidth=2.0)
    plt.plot(train_hist, label='Train (Test Samples) Error', linewidth=2.0)
    plt.legend(loc='best')
    # plt.yscale('log')
    plt.grid(True)
    plt.show()

# Visual Evaluation
test_eval(evl_func, X_tst, shape=(-1, img.shape[2],
                                  WINDOW_SIZE * rescale_factor,
                                  WINDOW_SIZE * rescale_factor), show_all=False)

# Visualize
# show_network(ntwrk)
# plot_conv_weights(lasagne.layers.get_all_layers(ntwrk)[9])
# X_tst = set_divider(Xs, Ys)[4]  # Get the test data for a hones test :)
# encode = lasagne.layers.get_all_layers(ntwrk)[5]
# out = lasagne.layers.get_output(encode, inputs=X_tst).eval()
