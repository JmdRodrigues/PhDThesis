import numpy as np
import matplotlib.pyplot as plt

def second_derivative(s):
    """
    computes the second derivative of signal s
    :param s:
    :return:
    """
    ds = np.diff(s)
    ds = np.append(ds, ds[-1])
    dds = np.diff(ds)
    dds = np.append(dds, dds[-1])

    return dds

def quantilestates2(sample, quantile_vals, char_list):
    if(np.size(quantile_vals) == 1):
        sample_quantiles = [quantile_vals.tolist()]
    else:
        sample_quantiles = quantile_vals.tolist()

    sample_quantiles.append(sample)
    return char_list[list(np.sort(sample_quantiles)).index(sample)]

def quantilstatesArray2(signal, quantile_vals, char_list, conc=True):
    if(conc):
        return "".join([quantilestates2(sample, quantile_vals, char_list) for sample in signal])
    else:
        out_str = [quantilestates2(sample, quantile_vals, char_list) for sample in signal]

        return out_str


def max_AmpChange(s, m):
    """

    :param s: signal
    :param m: list of list of matches
    :return: value of maximum amplitude difference
    """

    max_ampdif = 0


    for matches in m:
        for match in matches:
            if(max_ampdif<abs(s[match[0]-1] - s[match[1]-1])):
                 max_ampdif = abs(s[match[0]-1] - s[match[1]-1])


    return max_ampdif


def get_dif_segments(s, thr):

    ds = np.abs(second_derivative(s))

    ds_peaks = detect_peaks(ds, mph=thr*max(ds))
    ds_match = [(a, b) for a, b in zip(ds_peaks[:-1], ds_peaks[1:])]
    ds_match.append((ds_match[-1][-1], len(s)))
    ds_match.insert(0, (0, ds_match[0][0]))

    return ds_match

def get_slope_segments(s, thr=0.2):
    matches = get_dif_segments(s, thr)
    p_l = []
    for i, match in enumerate(matches):
        segment = s[match[0]:match[1]]
        slope_i = (segment[-1]-segment[0])/len(segment)
        p_l.append([slope_i]*len(segment))

    # plt.plot([match[0] for match in matches], s[[match[0] for match in matches]], 'o')
    # plt.plot(np.abs(500*np.concatenate(p_l)))
    # plt.plot(s)
    # plt.show()

    return np.concatenate(p_l)

def abs_distance_2_linear(segment):
    x = np.linspace(0, len(segment), len(segment))
    z = np.polyfit(x, segment, 1)
    p = np.poly1d(z)
    linear_apprx = p(x)

    return np.sum(np.abs(linear_apprx-segment))

def distance_2_linear(segment):
    x = np.linspace(0, len(segment), len(segment))
    z = np.polyfit(x, segment, 1)
    p = np.poly1d(z)
    linear_apprx = p(x)

    return np.sum(linear_apprx-segment)


def get_difference_between_linear_segment(s, thr=0.2):
    """
    Distances to a linear approximation of segments that are separated based on the
    peaks of the absolute second derivative.
    :param s: signal
    :return: distances per segment, with the same size as the original signal
    """
    matches = get_dif_segments(s, thr)
    distances_ = []
    for i, match in enumerate(matches):
        segment = s[match[0]:match[1]]
        # calculate distances to fitted data
        distances_.append([distance_2_linear(segment)]*len(segment))

    plt.plot([match[0] for match in matches], s[[match[0] for match in matches]], 'o')
    plt.plot(np.abs(np.concatenate(distances_)))
    plt.plot(s)
    plt.show()

    return np.concatenate(distances_)

def get_concavity_per_segment(s, thr=0.2):
    """
    Computes the concavity of the signal by means of the second derivative.
    When the second derivative is negative, the segment is concave, and when positive, the segment is convex
    :param s:
    :return:
    """
    matches = get_dif_segments(s, thr)
    dds = second_derivative(s)

    concavity = []

    for i, match in enumerate(matches):
        ddsegment = dds[match[0]:match[1]]
        concavity.append([np.sign(np.sum(np.sign(ddsegment)))]*len(ddsegment))

    plt.plot([match[0] for match in matches], s[[match[0] for match in matches]], 'o')
    plt.plot((np.concatenate(concavity)))
    plt.plot(s)
    plt.show()

    return np.concatenate(concavity)

def get_concavity_absolute(s):
    dss = second_derivative(s)

    return np.sign(dss)

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb"""


    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
