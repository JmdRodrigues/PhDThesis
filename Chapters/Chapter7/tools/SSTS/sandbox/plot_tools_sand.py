from tools.plot_tools import Cplot
import numpy as np
import matplotlib.pyplot as plt

# def plt_fill_textcolorized(signal, str_signal, ax):
#     Cplot(signal, ax=ax)
#     for i, char in enumerate(set(str_signal)):
#         condition = np.array([char_i == char for char_i in str_signal])
#         ax.fill_between(np.linspace(0, len(signal), len(signal)), 0, signal, where=condition, color=color_list[i])