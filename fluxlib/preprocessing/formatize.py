import numpy as np
import pandas as pd
from datetime import datetime

class Formatter():
    def __init__(self, p, skiprows = None, nodata = None):
        self.skiprows = skiprows
        self.nodata = nodata
        self._df = self.load_csv(p)
        
    def __call__(self, index_col = 0, fmt = "%Y-%m-%d %H:%M:%S", renames = {}):
        if hasattr(self, "df"):
            df = self.df.copy()
        else:
            df = self._df.copy()
        if not isinstance(index_col, str):
            index_col = df.columns[int(index_col)]
        df = df.set_index(index_col, drop = True)
        df.index = df.index.map(
            lambda x: datetime.strptime(x, fmt)
        )
        if renames:
            df = df[list(renames.keys())]
            df = df.rename(columns = renames)
        return df

        @property
        def df(self):
            """
            Pythonic getter
            """
            return self._df

        @df.setter
        def df(self, value):
            """
            Pythonic setter
            """
            self._df = value

        @df.deleter
        def df(self):
            """
            Pythonic deleter
            """
            del self._df
        
    def load_csv(self, p):
        df = pd.read_csv(p, skiprows = self.skiprows)
        df.columns = df.columns.str.strip()
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis = 1)
        if self.nodata:
            df = df.replace(self.nodata, np.nan)
        return df

    def customized_function(self, func, *args, **kwargs):
        if hasattr(self, "df"):
            df = self.df.copy()
        else:
            df = self._df.copy()
        df = func(df, *args, **kwargs)
        return df

    def datetime_from_concat_rows(self):
        # format YYYY-mm-dd HH:MM
        if hasattr(self, "df"):
            df = self.df.copy()
        else:
            df = self._df.copy()
        df["datetime"] = df["Year"].map(str) + "-" + \
            df["Month"].map(str) + "-" + \
            df["Day"].map(str) + " " + \
            df["Hour"].map(str) + ":" + \
            df["Minute"].map(str)
        return df