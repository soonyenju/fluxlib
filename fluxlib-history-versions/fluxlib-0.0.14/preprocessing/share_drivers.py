## Share drivers with other EC sites
import numpy as np
import pandas as pd
from datetime import datetime

def borrow(df_src, lender_path, flux, scale = "15T"):
    df = pd.read_csv(lender_path, index_col = 0)
    df.index = df.index.map(
        lambda x: datetime.strptime(x, r"%Y-%m-%d %H:%M:%S")
    )

    df = df.drop(flux, axis = 1)

    # upsampling
    upsampled = df_src.resample(scale)
    interpolated = upsampled.interpolate(method = "spline", order = 2)
    # intersecting two indices
    new_index = interpolated.index.intersection(df.index)
    df_trg = pd.concat([interpolated.loc[new_index, :], df.loc[new_index, :]], axis = 1)
    return df_trg