import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
import lasagne
import nolearn.lasagne.visualize
from itertools import product
from skimage import transform as tf


# Plots img array with shape (imgs, channel, width, height)
def plot_agumented_images(imgs, title=''):
    shape = imgs.shape
    nrows = np.ceil(np.sqrt(imgs.shape[0])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows, ncols, squeeze=False)
    figs.suptitle(title)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[0]:
            break
        axes[r, c].imshow(np.transpose(imgs[i], (2, 1, 0)), cmap='gray',
                          interpolation='none')


# Creates a gaussian matrix with given size
def make_gaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


# Theano & Lasagne Utils

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Give test function test samples and original output, plots all of them
def test_eval(eval_func, ass, shape, show_all=False):
    if not show_all:
        step_size = int(np.sqrt(ass.shape[0]))  # Show only sqrt(size) items
    else:
        step_size = 1
    selected_imgs = np.asarray([ass[i] for i in np.arange(0, ass.shape[0], step=step_size)])
    outputs = eval_func(selected_imgs)
    outputs = outputs[0][:]
    outputs = np.reshape(outputs, shape)
    plot_agumented_images(outputs, title='Transformed')
    plot_agumented_images(selected_imgs, title='Augmented')


def show_network(network):
    nolearn.lasagne.visualize.draw_to_file(lasagne.layers.get_all_layers(network),
                                           'network.png')
    img = imread('network.png')
    plt.figure()
    plt.imshow(img)


# This function divides the set given ratios
def set_divider(x_set, y_set):
    X_len = x_set.shape[0]
    X_tr = x_set[0:int(X_len*.7)]
    X_tst = x_set[int(X_len*.7): X_len - int(X_len*0.2)]
    X_val = x_set[X_len - int(X_len*0.2): X_len]

    Y_len = y_set.shape[0]
    Y_tr = y_set[0:int(Y_len*.7)]
    Y_tst = y_set[int(Y_len*.7): Y_len - int(Y_len*0.2)]
    Y_val = y_set[Y_len - int(Y_len*0.2): Y_len]
    return X_tr, Y_tr, X_val, Y_val, X_tst, Y_tst


# Resizer
def rescaler(Xs, Ys, rescale_factor):
    Xs = np.asarray([tf.rescale(np.transpose(Xs[i], (2, 1, 0)), scale=rescale_factor) for i in range(0, Xs.shape[0])])
    Xs = Xs.transpose((0, 3, 2, 1))
    Ys = np.asarray([tf.rescale(np.transpose(Ys[i], (2, 1, 0)), scale=rescale_factor) for i in range(0, Ys.shape[0])])
    Ys = Ys.transpose((0, 3, 2, 1))
    return Ys, Xs


# Reconstruction error
def compute_reconstruction_error(inputs, targets, outputs):
    """
    This function computes the reconstruction error.
    Metric 1 : Normalized rms structural transformation error : Each inputs distance to the target
    (targets are assumed to be fixed) will be calculated & summed. The final value is the mean of this summation
    This metric is normalized over target image

    Metric 2 : Normalized rms Regression error : Each outputs distance to the target will be calculated & summed.
    The final value is the mean of this summation.

    :param inputs: The augmented samples 4D Tensor
    :param targets: The target (un transformed image) 4D Tensor, the size equals to inputs, but all the same
    :param outputs: The output of the network (regression result) 4D tensort
    :return: reconstruction error
    """
    # Make input also linear so element wise division can be performed
    inputs_res = T.reshape(inputs, (inputs.shape[0], -1))

    # Metric 1
    mu_m1 = T.sum((inputs_res - targets) ** 2)
    mu_m1 /= targets.sum(axis=None) ** 2
    mu_m1 = T.sqrt(mu_m1)

    # Metric 2
    mu_m2 = T.sum((outputs - targets) ** 2)
    mu_m2 /= targets.sum(axis=None) ** 2
    mu_m2 = T.sqrt(mu_m2)

    # mu_m1 = T.mean((inputs_res - targets)**2)
    # mu_m2 = T.mean((outputs - targets)**2)

    # Ratio of two metrics, highest expectation is mu_m1
    return mu_m2 / mu_m1
