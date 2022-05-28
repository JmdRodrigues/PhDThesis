from colorutils import Color, ArithmeticModel
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def libphys_cmap():
    top = cm.get_cmap('Oranges_r', 128)
    print(top)
    bottom = cm.get_cmap('Blues', 128)
    print(bottom)
    newcolors = np.vstack((top(np.linspace(0.3, 1, 128)),
                           bottom(np.linspace(0, 0.8, 128))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')

    return newcmp

def libphys_cmap_r():
    bottom = cm.get_cmap('Oranges', 128)

    top = cm.get_cmap('Blues_r', 128)

    newcolors = np.vstack((top(np.linspace(0.3, 1, 128)),
                           bottom(np.linspace(0, 0.8, 128))))
    newcmp = ListedColormap(newcolors, name='BlueOrange')

    return newcmp

def mycmap():
    colors = ["maroon", "firebrick", "indianred", "rosybrown", "darksalmon", "coral", "orangered", "darkorange", "orange", "turquoise", "darkturquoise", "lightseagreen", "cornflowerblue", "royalblue", "darkblue", "darkslateblue", "slateblue", "mediumpurple"]
    newcmp = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

    return newcmp

def mycmap2():
    colors = ["indianred", "orange", "turquoise", "royalblue", "slateblue"]
    newcmp = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

    return newcmp

def mycmap3():
    colors = ["orangered", "royalblue"]
    newcmp = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

    return newcmp

def phd_thesis_cmap1():
    colors = ["seagreen", "turquoise", "lightcyan", "lavender", "deepskyblue", "royalblue"]
    newcmp = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

    return newcmp

color_list = ["dodgerblue", "orangered", "lightgreen", "mediumorchid", "gold", "firebrick", "darkorange", "springgreen", "lightcoral"]
# color_list = ["dodgerblue", "orangered", "lightgreen", "mediumorchid", "gold", "firebrick", "darkorange", "springgreen", "lightcoral", "violet", "crimson", "palegreen", "coral", "rosybrown", "aquamarine", "moccasin", "seagreen", "lavender"]
color_list2 = list(mcolors.CSS4_COLORS)
primary_colors = {"d":Color(web="#e3c44c", arithmetic=ArithmeticModel.BLEND), "a":Color(web="#d73824", arithmetic=ArithmeticModel.BLEND), "c":Color(web="#66CB5E", arithmetic=ArithmeticModel.BLEND), "b":Color(web="#6e91ee",arithmetic=ArithmeticModel.BLEND)}
#

