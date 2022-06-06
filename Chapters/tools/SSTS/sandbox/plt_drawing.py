import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import scipy as sc
import novainstrumentation as ni

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x_pts = [0]
        self.y_pts = [0]
        self.click = 0
        self.line, = self.ax.plot(self.x_pts, self.y_pts, marker="o")
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.connection = 0
        self.ax.figure.canvas.mpl_connect('button_press_event', self.start)
        # self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def start(self, event):
        if(self.click==0):
            self.connection = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_press)
            self.click += 1
        else:
            self.connection = self.ax.figure.canvas.mpl_disconnect(self.connection)

            return {"x":self.x_pts, "y":self.y_pts}

    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.x_pts.append(self.x0)
        self.y_pts.append(self.y0)
        self.line.set_xdata(self.x_pts)
        self.line.set_ydata(self.y_pts)
        self.ax.figure.canvas.draw()


class Annotate2(object):
    def __init__(self):
        self.ax = plt.gca()
        self.x_pts = []
        self.y_pts = []
        self.click = 0
        self.line, = self.ax.plot(self.x_pts, self.y_pts, marker="o")
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.connection = 0
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)


    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.x_pts.append(self.x0)
        self.y_pts.append(self.y0)
        self.line.set_xdata(self.x_pts)
        self.line.set_ydata(self.y_pts)
        self.ax.figure.canvas.draw()


class AnnotateSpan(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x_pts = [0]
        self.y_pts = [0]
        self.click = 0
        self.line, = self.ax.plot(self.x_pts, self.y_pts, marker="o")
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.connection = 0
        self.ax.figure.canvas.mpl_connect('button_press_event', self.start)
        # self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def start(self, event):
        if(self.click==0):
            self.connection = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_press)
            self.click += 1
        else:
            self.connection = self.ax.figure.canvas.mpl_disconnect(self.connection)

            return {"x":self.x_pts, "y":self.y_pts}

    def on_press(self, event):
        print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.x_pts.append(self.x0)
        self.y_pts.append(self.y0)
        self.line.set_xdata(self.x_pts)
        self.line.set_ydata(self.y_pts)
        self.ax.figure.canvas.draw()

def get_ts_drawing0():
    a = Annotate()
    plt.show()
    print(a.x_pts)
    f = sc.interpolate.interp1d(a.x_pts, a.y_pts)
    x_new = np.linspace(0, max(a.x_pts), len(a.x_pts))
    y_new = f(x_new)
    print(y_new)
    y_new = ni.smooth(y_new, len(y_new)//5)
    plt.plot(x_new, y_new)
    plt.plot(x_new, y_new, 'o')
    plt.show()

    return x_new, y_new

def get_ts_drawing():
    a = Annotate2()
    plt.show()
    return a.x_pts, a.y_pts

def get_ts_span_drawing():
    """
    TODO...
    :return:
    """
    a = AnnotateSpan()
    plt.show()
    return a.x_pts, a.y_pts

