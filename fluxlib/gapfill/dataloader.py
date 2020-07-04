import numpy as np
import pandas as pd
from datetime import datetime

class Loader():
    def __init__(self, data_path):
        self.data_path = data_path

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