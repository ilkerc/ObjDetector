import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Annotate(object):
    def __init__(self, cb, img_data=None, window_size=50):
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.cb = cb
        self.img_data = img_data
        self.window_size = window_size
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()
        img_bbx = self.img_data[int(self.y0):int(self.y1),
                                int(self.x0):int(self.x1),
                                :]
        self.cb(img_bbx)
