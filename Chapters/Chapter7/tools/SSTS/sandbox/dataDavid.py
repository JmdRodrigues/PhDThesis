from scipy.io import loadmat
from tools.plot_tools import Cplot
from tools.processing_tools import notchFilter, plotfft
from tools.plot_tools import plot_textcolorized
from novainstrumentation import bandpass
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from GrammarofTime.SSTS.sandbox.connotation_sandbox import AmplitudeTrans, AmpChange, D1Speed, SignConnotation, addArrayofStrings
import string
import time
from novainstrumentation import smooth


def ecg_preprocess(sig):
    sig_sm1 = smooth(sig, window_len=20)
    sig_sm = smooth(sig_sm1, window_len=500)
    sig_sm = sig_sm1-sig_sm
    sig_n = notchFilter(sig_sm, 60, 250, Q=5)
    sig_m = (sig_n-np.mean(sig_n))/np.max(sig_n)

    return sig_m


def Connotation2(sig):
    # amp_level = AmplitudeTrans(sig, 2, string.ascii_uppercase, method="quantiles")

    t0 = time.time()
    ampdiff_str = AmpChange(sig, 0.75, "absolute")

    ax1 = plt.subplot(1, 1, 1)
    # plot_textcolorized(wave3, wave3_conc_str_tpl[2], ax1)
    plot_textcolorized(sig, ampdiff_str, ax1)
    plt.show()
    t1 = time.time()

    print("Done with ampdiff...")
    print("time: " + str(t1 - t0))
    speed_str = D1Speed(sig, 0.75)
    t2 = time.time()
    print("Done with diff...")
    print("time: " + str(t2-t1))
    sign_str = SignConnotation(sig)
    t3 = time.time()
    print("Done with sign...")
    print("time: " + str(t3 - t2))
    print("creating string...")
    wave_str = addArrayofStrings([sign_str, ampdiff_str, speed_str])

    print("Done")

    return wave_str


path = "D:\PhD\Code\GrammarofTime\Data"
file = "\\fantasia_1o01m.mat"

#channel has the correct signal morphology
ecg = loadmat(path+file)["val"][1]
ecg_clean = ecg_preprocess(ecg)

plt.plot(ecg_clean)
plt.show()

ecg_str = Connotation2(ecg_clean)


