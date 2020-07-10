import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from scitbx import Yaml, create_all_parents
from pathlib import Path

class Formatter():
    def __init__(self, config_path):
        self.cfg = Yaml(config_path).load()

    def __call__(self):
        source = Path(self.cfg["fluxdata"]["source"])
        if self.cfg["fluxdata"]["is_file"]:
            source = [source]
        else:
            source = source.glob("*.csv")
        for src in source:
            self.name = src.stem
            print(self.name)
            df_flux = self.load_fluxdata(src)
            if "ERA5" in self.cfg:
                met = self.load_era5(self.cfg["ERA5"]["path"], *self.cfg["ERA5"]["params"])
                if "ssr" in self.cfg["ERA5"]["params"]:
                    met = self.ssr2rg(met)
                # insert ERA5 into fluxdata
                df_met = met.loc[df_flux.index, :]
                df_flux = pd.concat([df_flux, df_met], axis = 1)
            self.save_csv(df_flux)
            try:
                self.save_mds_txt(df_flux)
            except KeyError as e:
                print(e)
        # return df_flux

    def load_fluxdata(self, src, undef_qcNEE = 0): 
        if "skiprows" in list(self.cfg["fluxdata"].keys()):
            skiprows = [self.cfg["fluxdata"]["skiprows"]]
        else:
            skiprows = None
        try:
            df_raw = pd.read_csv(src, skiprows = skiprows)
            if self.cfg["fluxdata"]["index"] == "YmdHM":
                df_raw["YmdHM"] = df_raw["Year"].map(str) + "-" + \
                    df_raw["Month"].map(str) + "-" + \
                    df_raw["Day"].map(str) + " " + \
                    df_raw["Hour"].map(str) + ":" + \
                    df_raw["Minute"].map(str)
            elif self.cfg["fluxdata"]["index"] == "YjHM":
                pass
            df_raw = df_raw.set_index(
                df_raw[self.cfg["fluxdata"]["index"]].astype(str),
            )
            df_raw = df_raw.drop(self.cfg["fluxdata"]["index"], axis = 1)
        except Exception as e:
            # print(e)
            df_raw = pd.read_csv(src, index_col = 0, skiprows = skiprows)
        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop("Unnamed: 0", axis = 1)
        df_raw.index.name = "Datetime"
        # print(df_raw)

        df_raw.columns = df_raw.columns.str.strip()
        df_raw.index = df_raw.index.map(
            lambda x: datetime.strptime(x, self.cfg["time_format"])
        )
        df_raw = df_raw.drop_duplicates()
        df_raw = df_raw.replace(-9999, np.nan)

        # set default qcNEE if there is none.
        if not "qcNEE" in list(self.cfg["fluxdata"]["load_params"].keys()):
            df_raw["qcNEE"] = undef_qcNEE
            df = df_raw[
                list(
                    self.cfg["fluxdata"]["load_params"].values()
                    ) + ["qcNEE"]
                ]
        else:
            df = df_raw[list(self.cfg["fluxdata"]["load_params"].values())]
        del(df_raw)
        return df.rename(columns = {
            value:key for key, value in self.cfg["fluxdata"]["load_params"].items()
        })
    
    def save_csv(self, df, savefolder = None, savefile = None):
        if not savefile:
            savefile = self.name
        if not savefolder:
            savefolder = Path(self.cfg["output"]["folder"])
        else:
            savefolder = Path(savefolder)
        create_all_parents(savefolder)
        df.to_csv(Path(savefolder).joinpath(f"{savefile}_fmt.csv"))

    def save_mds_txt(self, df, savefolder = None, savefile = None):
        df["Year"] = df.index.map(
            lambda x: x.year
        )
        df["DoY"] = df.index.map(
            lambda x: np.int(x.strftime('%j'))
        )
        df["Hour"] = df.index.map(
            lambda x: x.minute / 60 + x.hour
        )

        df = df[["Year", "DoY", "Hour", "NEE", "Rg", "Tair", "VPD"]]
        df = df.reset_index(drop = True)
        df.loc[-1] = ["-", "-", "-", "umolm-2s-1", "Wm-2", "degC", "hPa"]
        df.index = df.index + 1  # shifting index
        df = df.sort_index()  # sorting by index

        if not savefile:
            savefile = self.name
        if not savefolder:
            savefolder = Path(self.cfg["output"]["folder"])
        else:
            savefolder = Path(savefolder)
        create_all_parents(savefolder)

        df = df.fillna(-9999.)

        df.to_csv(
            savefolder.joinpath(f"{savefile}_fmt.txt"), 
            index=None, 
            sep='\t', 
            mode='w'
        )

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
