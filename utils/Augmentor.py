import numpy as np
from matplotlib.pyplot import gca
from skimage import transform as tf
from ImageAugmenter import ImageAugmenter
from skimage.exposure import rescale_intensity


class Augmentor(object):
    def __init__(self, cb, img_data=None,
                 window_size=100,
                 degree_range=(-180, 181),
                 d_step=1,
                 count=100):
        self.ax = gca()
        self.cb = cb
        self.img_data = img_data
        self.count = count
        self.d_step = d_step
        self.degree_range = degree_range
        self.window_size = window_size
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

    @staticmethod
    def augmentor(h, w, img, count, scale_factor=1.0):
        agu = ImageAugmenter(img_height_px=img.shape[0],
                             img_width_px=img.shape[1],
                             scale_to_percent=1.2,
                             rotation_deg=(-180, 180),
                             shear_deg=(-20, 20),
                             translation_x_px=5,
                             translation_y_px=5,
                             transform_channels_equally=True)
        copy_imgs = np.asarray([img for _ in range(0, count)])
        return agu.augment_batch(copy_imgs)

    def rotate_img(self, img, step=1):
        return np.asarray([tf.rotate(img, degree) for degree in range(self.degree_range[0],
                                                                      self.degree_range[1],
                                                                      step)])

    def manuel(self, coords):
        return self.on_press(x_m=coords[0], y_m=coords[1])

    def on_press(self, event=None, x_m=None, y_m=None):
        center_of_crop = self.window_size * 2
        # If the method has been voked manuelly
        if event is None:
            x = x_m
            y = y_m
        else:
            y = int(event.xdata)
            x = int(event.ydata)

        # We'll crop the an area of 2x window size than apply transformation
        # Than crop it again from the agumented sample
        img_data_crop = self.img_data[x - self.window_size:x + self.window_size,
                        y - self.window_size:y + self.window_size]

        # Generate rotated versions &
        # Transpose augmented imgs for network (batch_size, channel, width, height)
        # augmented_imgs = self.rotate_img(img_data_crop, step=self.d_step)
        augmented_imgs = Augmentor.augmentor(center_of_crop, center_of_crop, img_data_crop, self.count)
        augmented_imgs = np.transpose(augmented_imgs, (0, 3, 2, 1))
        aug_size = augmented_imgs.shape[0]

        # Crop the images here, while we don't want to see the black sides
        batch_shape = (aug_size, self.img_data.shape[2], self.window_size, self.window_size)
        cropped_original = np.zeros(shape=batch_shape)
        cropped_augmented = np.zeros(shape=batch_shape)
        start = int(center_of_crop * (1 / 4.))
        stop = int(center_of_crop * (3 / 4.))

        img_data_crop = img_data_crop / 255.
        for i in range(0, aug_size):
            cropped_original[i] = np.transpose(img_data_crop[start:stop, start:stop, :], (2, 1, 0))
            cropped_augmented[i] = augmented_imgs[i, :, start:stop, start:stop]

        # Here are the return values
        if event is None:
            return cropped_original, cropped_augmented
        else:
            self.cb(cropped_original, cropped_augmented)
