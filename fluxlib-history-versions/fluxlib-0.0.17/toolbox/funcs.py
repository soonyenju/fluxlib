import numpy as np
import pandas as pd
from datetime import datetime

def set_mds_txt(df, savefile, flux = "NEE", fluxunit = "umolm-2s-1", driver_units = {}):
    """
    index should be datetime type
    """
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
    units = ["-", "-", "-", fluxunit, "Wm-2", "degC", "hPa"]
    if driver_units:
        for (d, u) in driver_units.items():
            drivers.append(d)
            units.append(u)
    df = df[drivers]
    df = df.reset_index(drop = True)
    df.loc[-1] = units
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index

    df = df.fillna(-9999.)

    df.to_csv(
        savefile, 
        index=None, 
        sep='\t', 
        mode='w'
    )