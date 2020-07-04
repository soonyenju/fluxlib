import sys
import numpy as np

def make_gaps(pointers, window_size, rgap, series, rbak = 3, vtheta = 0.5):
    length = len(series)
    intersect = np.intersect1d(pointers, pointers - (window_size - 1))

    n_gap = np.int(length * rgap / window_size)

    anchors = np.random.choice(intersect, np.ceil(n_gap * rbak).astype(np.int))

    samples = []
    count = 0
    for idx, anc in enumerate(anchors):
        sample = np.arange(anc, anc + window_size)
        tmp_series = series.iloc[sample, :]
        if np.isin(sample, intersect).all(): # sample must all in the inersect
            if (idx > 0) and (np.min(np.abs(anchors[0: idx] - anc)) < window_size): # anc should be far from recorded anchors
                continue
            if len(tmp_series.dropna()) / len(tmp_series) < vtheta:
                continue
            pointers = np.setdiff1d(pointers, sample)
            samples.extend(sample.tolist())
            count += 1
        if count >= n_gap:
            break
    return samples, pointers