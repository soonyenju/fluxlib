import numpy as np
import pandas as pd
from datetime import datetime

class Loader():
    def __init__(self, data_path):
        self.data_path = data_path

    def __call__(self, timestamp_format = r"%Y-%m-%d %H:%M:%S", nodata = None):
        # load formatted csv data
        df = pd.read_csv(self.data_path, index_col = 0)
        if (type(df.index[0]) == pd._libs.tslibs.timestamps.Timestamp):
            pass
        else:
            df.index = df.index.map(
                lambda x: datetime.strptime(str(x), timestamp_format)
            )
        if nodata:
            df = df.replace(nodata, np.nan)
        return df

    # load and format fluxnet EC data
    def load_fluxnet(self, specified_columns, timestamp_format):
        #----------------------------------------------
        # load data
        raw_df = pd.read_csv(self.data_path)
        raw_df = raw_df.replace(-9999, np.nan)
        #-----------------------------------------------
        # set timestamp as index
        raw_df = raw_df.set_index(
            raw_df["TIMESTAMP_START"].map(
                lambda x: datetime.strptime(str(x), timestamp_format)
            )
        )
        # extract specified columns
        df = raw_df.loc[:, specified_columns]
        #-----------------------------------------------
        return df

    # load and format malaysia EC data
    def load_lowcost(self, specified_columns, timestamp_format):
        #----------------------------------------------
        # load data
        raw_df = pd.read_csv(self.data_path, skiprows = [1])
        raw_df = raw_df.replace(-9999, np.nan)
        #----------------------------------------------
        # covert timestamp to datetime object
        timestamp = raw_df["Year"].map(str) + "-" + \
                    raw_df["Month"].map(str) + "-" + \
                    raw_df["Day"].map(str) + " " + \
                    raw_df["Hour"].map(str) + ":" + \
                    raw_df["Minute"].map(str)
        #-----------------------------------------------
        # extract specified columns
        df = raw_df.loc[:, specified_columns]
        #-----------------------------------------------
        # set timestamp as index
        df = df.set_index(
            timestamp.map(
                lambda x: datetime.strptime(x, timestamp_format)
            )
        )
        return df

    # load standardized format EC
    def load_format(self, specified_columns, timestamp_format):
        #----------------------------------------------
        # load data
        raw_df = pd.read_csv(self.data_path, index_col = 0)
        raw_df = raw_df.replace(-9999, np.nan)
        #-----------------------------------------------
        # # set timestamp as index
        raw_df.index = raw_df.index.map(
            lambda x: datetime.strptime(str(x), timestamp_format)
        )
        # raw_df.index = pd.to_datetime(raw_df.index)
        # extract specified columns
        df = raw_df.loc[:, specified_columns]
        #-----------------------------------------------
        return df