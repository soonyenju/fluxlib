import sys
import numpy as np
from pathlib import Path
from scitbx import Yaml, create_all_parents

def make_gap_pipeline(config_path, df, flux, rbak = 3, vtheta = 0.5, seed = 0):
    """
    rbak: r-fold backup gap beginning points 
    vtheta: valid half-hours/(artificial gap - existing gap)
    """
    cfg = Yaml(config_path).load()
    if not isinstance(flux, list):
        flux = [flux]
    series = df[flux]

    np.random.seed(seed)
    pointers = np.arange(len(series))
    samples = []
    tags = []
    for batch_cnt, gap in enumerate(cfg["gaps"]):
        # print(gap)
        window_size = gap["window_size"]
        p = cfg["tgr"] * gap["ratio"]
        sample, pointers = make_gaps(pointers, window_size, p, series, rbak = rbak, vtheta = vtheta)
        samples.extend(sample)
        tags.extend(list(np.ones(len(sample)) * batch_cnt))

    train_idxs = pointers
    test_idxs = np.array(samples)
    tags = np.array(tags)
    # remove NaNs (existing gaps) in both train and test indices:
    train_idxs = train_idxs[np.where(np.isfinite(series.iloc[train_idxs]))[0]]
    valid_test_idxs_loc = np.where(np.isfinite(series.iloc[test_idxs]))[0]
    test_idxs = test_idxs[valid_test_idxs_loc]
    tags = tags[valid_test_idxs_loc]

    return train_idxs, test_idxs, tags

def make_gaps(pointers, window_size, rgap, series, rbak = 3, vtheta = 0.5):
    """
    rgap: gap ratio
    """
    length = len(series)
    # valid gap beginning points:
    intersect = np.intersect1d(pointers, pointers - (window_size - 1))

    # gap number:
    n_gap = np.int(length * rgap / window_size)

    # gap beginning points pool
    anchors = np.random.choice(intersect, np.ceil(n_gap * rbak).astype(np.int))

    samples = []
    count = 0
    for idx, anc in enumerate(anchors):
        sample = np.arange(anc, anc + window_size)
        tmp_series = series.iloc[sample, :]
        # sample must all in the inersect
        if np.isin(sample, intersect).all():
            # anc should be far from recorded anchors, avoid overlapping
            if (idx > 0) and (np.min(np.abs(anchors[0: idx] - anc)) < window_size): 
                continue
            if len(tmp_series.dropna()) / len(tmp_series) < vtheta:
                continue
            pointers = np.setdiff1d(pointers, sample)
            samples.extend(sample.tolist())
            count += 1
        if count >= n_gap:
            break
    return samples, pointers


def save_mds_txt(df, test_idxs, savefolder, savefile, flux = "NEE", flux_unit = "umolm-2s-1", driver_units = {}):
    df.loc[df.index[test_idxs], flux] = -9999
    df["Year"] = df.index.map(
        lambda x: x.year
    )
    df["DoY"] = df.index.map(
        lambda x: np.int(x.strftime('%j'))
    )
    df["Hour"] = df.index.map(
        lambda x: x.minute / 60 + x.hour
    )
    
    drivers = ["Year", "DoY", "Hour", flux, "Rg", "Tair", "VPD"]
    units = ["-", "-", "-", flux_unit, "Wm-2", "degC", "hPa"]
    if driver_units:
        for (d, u) in driver_units.items():
            drivers.append(d)
            units.append(u)
    df = df[drivers]
    df = df.reset_index(drop = True)
    df.loc[-1] = units
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index

    savefolder = Path(savefolder)
    create_all_parents(savefolder)

    df = df.fillna(-9999.)

    df.to_csv(
        savefolder.joinpath(f"{savefile}_mds.txt"), 
        index=None, 
        sep='\t', 
        mode='w'
    )