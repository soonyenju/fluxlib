import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scitbx import Yaml
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

class GFiller():
    def __init__(self, config_path, seed = 0):
        np.random.seed(seed)
        warnings.simplefilter("ignore")
        self.cfg = Yaml(config_path).load()
        self.__get_regr__()
    
    def __get_regr__(self):
        regr_name = self.cfg["regressor"]
        if regr_name == "RFR":
            from sklearn.ensemble import RandomForestRegressor
            regr = RandomForestRegressor(**self.cfg["params"])
        elif regr_name == "XGB":
            from xgboost import XGBRegressor
            regr = XGBRegressor(**self.cfg["params"])
        elif regr_name == "DF21":
            from deepforest import CascadeForestRegressor
            regr = CascadeForestRegressor(**self.cfg["params"])
        elif regr_name == "GBR":
            from sklearn.ensemble import GradientBoostingRegressor
            regr = GradientBoostingRegressor(**self.cfg["params"])
        elif regr_name == "ABR":
            from sklearn.ensemble import AdaBoostRegressor
            regr = AdaBoostRegressor(**self.cfg["params"])
        elif regr_name == "MLP":
            from sklearn.neural_network import MLPRegressor
            regr = MLPRegressor(**self.cfg["params"])
        elif regr_name == "SVR":
            from sklearn.svm import SVR
            regr = SVR(**self.cfg["params"])
        else:
            raise Exception("No correctly specified regressor")
        self.regr = regr

    # new gap-filling method:
    #====================================================================================================
    # select_year start and end year
    @classmethod
    def select_year(self, start_year, end_year):
        if start_year == end_year:
            df = df[str(start_year)]
        else:
            df = df[str(start_year): str(end_year)]
        return df
    #====================================================================================================
    # set flux statistics
    @classmethod
    def set_stats(self, df, flux, scale = "15T"):
        flux_orig = df[flux].copy()
        df[flux] = df[flux].interpolate()#.fillna(method = "bfill")
        #-------------------------------------------------
        # set max
        flux_max = df[flux].resample("D").max()
        df["flux_max"] = flux_max.resample(scale).bfill()
        df["flux_max"] = df["flux_max"]#.ffill()
        #-------------------------------------------------
        # set min
        flux_min = df[flux].resample("D").min()
        df["flux_min"] = flux_min.resample(scale).bfill()
        df["flux_min"] = df["flux_min"]#.ffill()
        #-------------------------------------------------
        # set mean
        flux_mean = df[flux].resample("D").mean()
        df["flux_mean"] = flux_mean.resample(scale).bfill()
        df["flux_mean"] = df["flux_mean"]#.ffill()
        #-------------------------------------------------
        # set std
        flux_std = df[flux].resample("D").std()
        df["flux_std"] = flux_std.resample(scale).bfill()
        df["flux_std"] = df["flux_std"]#.ffill()
        #-------------------------------------------------
        # set 25%, 50%, 75% quantiles
        #----------------------------
        # 25%:
        flux_p25 = df[flux].resample("D").quantile(0.25)
        df["flux_p25"] = flux_p25.resample(scale).bfill()
        df["flux_p25"] = df["flux_p25"]#.ffill()
        #----------------------------
        # 50%:
        flux_p50 = df[flux].resample("D").quantile(0.50)
        df["flux_p50"] = flux_p50.resample(scale).bfill()
        df["flux_p50"] = df["flux_p50"]#.ffill()
        #----------------------------
        # 75%:
        flux_p75 = df[flux].resample("D").quantile(0.75)
        df["flux_p75"] = flux_p75.resample(scale).bfill()
        df["flux_p75"] = df["flux_p75"]#.ffill()

        df = df.interpolate()
        df[flux] = flux_orig
        return df, ["flux_max", "flux_min", "flux_mean", "flux_std", "flux_p25", "flux_p50", "flux_p75"]
    #====================================================================================================
    # set hour diff
    @classmethod
    def set_hour_diff(self, df):
        df["hour_dif"] = (df.index - df.index[0]).total_seconds() / 3600
        return df, ["hour_dif"]    
    #====================================================================================================
    # set season tag
    @classmethod
    def set_season_tag(self, df):
        df["season"] = (df.index.month%12 + 3) // 3 # print(seasons)
        return df, ["season"]
    #====================================================================================================
    # set radiance tag
    @classmethod
    def set_rg_tag(self, df, rg):
        df["rg_rank"] = np.select(
            condlist = [
                df[rg] < 10,
                (df[rg] > 10) & (df[rg] < 100),
                df[rg] > 100
            ],
            choicelist = [
                1,
                2,
                3
            ],
            default = 0
        )
        return df, ["rg_rank"]
    #====================================================================================================
    # set radiance DOY and year
    @classmethod
    def set_doy_year_tag(self, df):
        df["doy"] = df.index.map(
            lambda x: np.int(x.strftime("%j"))
        )
        df["year"] = df.index.year
        return df, ["doy", "year"]
    #====================================================================================================
    # auto-optimize RFR params
    @classmethod
    def auto_optimize(self, regr, X, y):
        from sklearn.model_selection import GridSearchCV
        # Create the parameter grid based on the results of random search 
        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
        }
        # regr is a based model
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = regr, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X, y)
        return grid_search.best_params_
    #====================================================================================================
    # train regressor:
    @classmethod
    def train(self, regr, X_train, y_train):
        # print(regr)
        regr.fit(X_train.values, y_train.values)
        return regr
    #====================================================================================================
    # test regressor
    @classmethod
    def test(self, regr, X_test, y_test, stat = True):
        predicts = regr.predict(X_test.values)
        if predicts.ndim == 1:
            predicts = predicts[:, np.newaxis]
        df = pd.DataFrame(np.concatenate([y_test, predicts], axis = 1), columns = ["truth", "estimates"])
        if stat:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()["truth"], df.dropna()["estimates"])
            r2 = r_value**2
            mse = mean_squared_error(df.dropna()["truth"], df.dropna()["estimates"])
            rmse = np.sqrt(mse)
            bias = np.mean(df.dropna()["estimates"] - df.dropna()["truth"])
            return df, r2, slope, rmse, bias
        else:
            return df
    #====================================================================================================
    # apply regressor
    @classmethod
    def apply(self, regr, X_apply, df, flux):
        predicts = regr.predict(X_apply.values)
        if predicts.ndim == 1:
            predicts = predicts[:, np.newaxis]

        filled_name = flux[0] + "_filled" # assume flux is one element list, e.g. ["NEE"]
        df[filled_name] = df[flux].copy()
        df.loc[X_apply.index, filled_name] = predicts
        return df[flux + [filled_name]]
    #====================================================================================================

    def run_filling_pipeline(self, df, itrain = None, itest = None, sitename = "test_site"):
        # itrain, itest: indices in df for training and testing, respectively
        drivers = self.cfg["drivers"]
        flux = self.cfg["flux"]
        rg = self.cfg["rg"]
        df = df[drivers + flux].copy()
        #-------------------------------------------------
        # set tags:
        df, stat_tags = self.set_stats(df, flux)
        df, season_tag = self.set_season_tag(df)
        df, rg_tag = self.set_rg_tag(df, rg)
        df, doy_year_tag = self.set_doy_year_tag(df)
        #-------------------------------------------------
        # prepare and split data for regressor
        param_columns = drivers + stat_tags + season_tag + rg_tag + doy_year_tag
        # X = df.dropna()[param_columns]
        # y = df.dropna()[flux]
        X = df[param_columns]
        y = df[flux]
        # # X_train, y_train, X_test, y_test are supposed to NOT contain any NaNs!
        # X = X.interpolate(method = "pad")
        # y = y.interpolate(method = "pad")
        if (not itrain is None) & (not itest is None):
            print("using input train & test")
            X_train = X.iloc[itrain, :]
            y_train = y.iloc[itrain]
            X_test = X.iloc[itest, :]
            y_test =y.iloc[itest]
        else:
            print("randomly split train & test")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size = 0.33, 
                random_state = 42
            )

        X_apply = df.loc[df[flux[0]].isna().index, param_columns].interpolate().bfill()
        #--------------------------------------------------
        # train and test/apply RFR
        regr = self.train(self.regr, X_train, y_train)
        result_df, r2, slope, rmse, bias = self.test(regr, X_test, y_test)
        result_df.index = df.index[itest]
        print(f"{sitename}, R2: {np.round(r2, 4)}, SLOPE: {np.round(slope, 4)}, RMSE: {np.round(rmse, 4)}, BIAS: {np.round(bias, 4)}")
        applied_df = self.apply(self.regr, X_apply, df, flux)
        return result_df, applied_df