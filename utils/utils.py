import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
import lasagne
import nolearn.lasagne.visualize
from itertools import product


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
    plot_agumented_images(selected_imgs, title='Agumented')


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
