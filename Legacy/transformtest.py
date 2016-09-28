import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.color import rgb2gray
from planar import BoundingBox, Affine

img = data.astronaut()
img_gray = rgb2gray(img)
plt.figure()
plt.subplot(231)
plt.title('Gray Image')
plt.imshow(img_gray, cmap='gray')

plt.subplot(232)
plt.title('RGB Image')
plt.imshow(img)

# This should be equivalent to:
width = 100
height = 100
x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                       np.linspace(-1, 1, height))
ones = np.ones(np.prod(x_t.shape))
grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])


x0 = 140
x1 = 310
y0 = 20
y1 = 180
img_crop = img_gray[y0:y1, x0:x1]
plt.subplot(233)
plt.title('Cropped Image')
plt.imshow(img_crop, cmap='gray')

bbox1 = BoundingBox([(x1, y1), (x0, y0)])
bbox2 = BoundingBox(bbox1 * Affine.rotation(15))

xx0 = 89
xx1 = 295
yy0 = 56
yy1 = 255
img_crop2 = img_gray[yy0:yy1, xx0:xx1]
plt.subplot(234)
plt.title('Rotated Image')
plt.imshow(img_crop2, cmap='gray')

# Create a grid
e_grid = np.zeros((y1-y0, x1-x0))
xx = np.linspace(x0, x1, x1-x0+1)
yy = np.linspace(y0, y1, y1-y0+1)
