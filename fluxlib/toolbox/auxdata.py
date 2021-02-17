import numpy as np
import pandas as pd
import xarray as xr

class Auxdata():
    def __init__(self):
        pass
    @classmethod
    def load_era5(self, era5_path, *args, scale = "30T"):
        with xr.open_dataset(era5_path) as ds:
            df = ds.to_dataframe()
            df.index = df.index.get_level_values("time")
            # print(df.columns)
        df = df[list(args)] # type(args): tuple
        df = df.resample(scale).mean().bfill()
        return df
    @classmethod
    def ssr2rg(self, df):
        # J/m-2 to W/m-2
        df["ssr"] = df["ssr"] / 3600
        df = df.rename(
            columns = {
                "ssr": "Rg"
            }
        )
        df[df["Rg"] < 0] = 0
        return df