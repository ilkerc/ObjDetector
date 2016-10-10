import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def transform(image, theta):
    image = np.expand_dims(image, axis=0).transpose((0, 3, 1, 2))
    theta = np.reshape(theta, (-1, 2, 3))
    num_batch, num_channel, height, width = image.shape

    out_height = height  # No Downsample
    out_width = width
    grid = meshgrid(out_height, out_width)

    T_g = np.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flatten = x_s.flatten()
    y_s_flatten = y_s.flatten()

    input_dim = np.transpose(image, (0, 2, 3, 1))
    input_transformed = interpolate(input_dim, x_s_flatten, y_s_flatten, out_height, out_width)
    return np.reshape(input_transformed, image.shape).transpose((0, 2, 3, 1)).squeeze()


def meshgrid(height, width):
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                           np.linspace(-1, 1, height))
    ones = np.ones(np.prod(x_t.shape))
    grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    return grid


def interpolate(im, x, y, out_height, out_width):
    num_batch, height, width, channels = im.shape
    height_f = float(height) # T.cast(height, theano.config.floatX)
    width_f = float(width)  # T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = np.clip(x, -1, 1)
    y = np.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = np.floor(x)
    y0_f = np.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = x0_f.astype(int)
    y0 = y0_f.astype(int)
    x1 = (np.minimum(x1_f, width_f - 1)).astype(int)
    y1 = (np.minimum(y1_f, height_f - 1)).astype(int)

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width * height
    base = np.repeat(
        np.arange(num_batch, dtype='int64') * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = np.expand_dims((x1_f - x) * (y1_f - y), axis=1)
    wb = np.expand_dims((x1_f - x) * (y - y0_f), axis=1)
    wc = np.expand_dims((x - x0_f) * (y1_f - y), axis=1)
    wd = np.expand_dims((x - x0_f) * (y - y0_f), axis=1)
    output = np.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id], axis=0)
    return output

"""
Test With Array =
m = np.zeros((100, 100, 3), float)
m.fill(0)
np.fill_diagonal(m[:,:,0], 120)
np.fill_diagonal(m[:,:,1], 120)
np.fill_diagonal(m[:,:,2], 120)
img = m
"""


img = mpimg.imread('/home/ilker/Desktop/img2.jpg')
the = np.array([1, 0, 0, 0, 1, 0])
transformed = np.asarray(transform(img, the), dtype='uint8')
plt.figure(); plt.imshow(img, interpolation="nearest")
plt.figure(); plt.imshow(transformed, interpolation="nearest")