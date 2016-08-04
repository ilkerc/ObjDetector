from __future__ import print_function
import theano
import lasagne
import matplotlib.pyplot as plt
import theano.tensor as T
import numpy as np
from utils.Augmentor import Augmentor
from utils.utils import iterate_minibatches, plot_agumented_images, test_eval
from Models import build_st_network, build_cnnae_network
from scipy import misc

# Constants
BATCH_SIZE = 30
WINDOW_SIZE = 64
# Global Variables
agumented_samples = None
original_samples = None


def train_network(x_data, y_data, num_epochs=100, batch_size=BATCH_SIZE, model='cnn'):
    # Make Y a matrix
    y_data = np.reshape(y_data, (y_data.shape[0], -1))
    # Building the model
    if model == 'cnn':
        network = build_cnnae_network(x_data.shape)
    else:
        network = build_st_network(x_data.shape)

    params = lasagne.layers.get_all_params(network, trainable=True)
    print("Model is ready for training")

    # Trainer Function and Variable Holders
    X = T.tensor4(dtype=theano.config.floatX)
    Y = T.matrix(dtype=theano.config.floatX)
    # Adaptive Learning Rate
    # learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
    # base_lr = 0.08
    # decay_lr = 0.95

    # Functions
    output = lasagne.layers.get_output(network, X, deterministic=False)
    cost = T.mean(lasagne.objectives.squared_error(output, Y))
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=0.1, momentum=0.9)
    train_func = theano.function([X, Y], [cost, output], updates=updates, allow_input_downcast=True)
    eval_func = theano.function([X], [output], allow_input_downcast=True)

    # Training Iterator
    print("Training Is About to Start")
    try:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            #  lr_for_epoch = base_lr * (decay_lr ** epoch)
            for batch in iterate_minibatches(x_data, y_data, batch_size, shuffle=False):
                inputs, targets = batch
                train_err += train_func(inputs, targets)[0]
                train_batches += 1
            print("Epoch {0}: Train cost {1}".format(epoch, train_err))
    except KeyboardInterrupt:
        pass
    print("Completed, saved")
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    return eval_func, train_func, network


# Callback from plot click
def selected_callback(os, ass):
    global agumented_samples
    global original_samples
    original_samples = os
    agumented_samples = ass
    print("Agumented Samples Has Been Created")

# Main Run
path = '/Users/ilker/Desktop/me.png'
img = misc.imread(path, flatten=False)  # Set flatten true if working w/ gray i mages
img = img[:, :, :3] / 255.  # Just in case if the image has more then 3 channels like alpha
a = Augmentor(selected_callback, img_data=img, window_size=WINDOW_SIZE, degree_range=(-45, 45), d_step=1)
plt.imshow(img)
plt.show()

evl_func, trn_func, ntwrk = train_network(agumented_samples,
                                          original_samples,
                                          num_epochs=200,
                                          batch_size=agumented_samples.shape[0],
                                          model='st')

test_eval(evl_func, agumented_samples, shape=(-1, img.shape[2], WINDOW_SIZE, WINDOW_SIZE), show_all=False)
