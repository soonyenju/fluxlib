import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

class Filler():
    def __init__(self, seed = 0):
        np.random.seed(seed)
        warnings.simplefilter("ignore")
    # new gap-filling method:
    #====================================================================================================
    # select_year start and end year
    @staticmethod
    def select_year(start_year, end_year):
        if start_year == end_year:
            df = df[str(start_year)]
        else:
            df = df[str(start_year): str(end_year)]
        return df
    #====================================================================================================
    # set flux statistics
    @staticmethod
    def set_stats(df, flux):
        #-------------------------------------------------
        # set max
        flux_max = df[flux].resample("D").max()
        df["flux_max"] = flux_max.resample('30T').bfill()
        df["flux_max"] = df["flux_max"].ffill()
        #-------------------------------------------------
        # set min
        flux_min = df[flux].resample("D").min()
        df["flux_min"] = flux_min.resample('30T').bfill()
        df["flux_min"] = df["flux_min"].ffill()
        #-------------------------------------------------
        # set mean
        flux_mean = df[flux].resample("D").mean()
        df["flux_mean"] = flux_mean.resample('30T').bfill()
        df["flux_mean"] = df["flux_mean"].ffill()
        #-------------------------------------------------
        # set std
        flux_std = df[flux].resample("D").std()
        df["flux_std"] = flux_std.resample('30T').bfill()
        df["flux_std"] = df["flux_std"].ffill()
        #-------------------------------------------------
        # set 25%, 50%, 75% quantiles
        #----------------------------
        # 25%:
        flux_p25 = df[flux].resample("D").quantile(0.25)
        df["flux_p25"] = flux_p25.resample('30T').bfill()
        df["flux_p25"] = df["flux_p25"].ffill()
        #----------------------------
        # 50%:
        flux_p50 = df[flux].resample("D").quantile(0.50)
        df["flux_p50"] = flux_p50.resample('30T').bfill()
        df["flux_p50"] = df["flux_p50"].ffill()
        #----------------------------
        # 75%:
        flux_p75 = df[flux].resample("D").quantile(0.75)
        df["flux_p75"] = flux_p75.resample('30T').bfill()
        df["flux_p75"] = df["flux_p75"].ffill()
        return df, ["flux_max", "flux_min", "flux_mean", "flux_std", "flux_p25", "flux_p50", "flux_p75"]
    #====================================================================================================
    # set season tag
    @staticmethod
    def set_season_tag(df):
        df["season"] = (df.index.month%12 + 3) // 3 # print(seasons)
        return df, ["season"]
    #====================================================================================================
    # set radiance tag
    @staticmethod
    def set_rg_tag(df, rg):
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
    @staticmethod
    def set_doy_year_tag(df):
        df["doy"] = df.index.map(
            lambda x: np.int(x.strftime("%j"))
        )
        df["year"] = df.index.year
        return df, ["doy", "year"]
    #====================================================================================================
    # auto-optimize RFR params
    @staticmethod
    def auto_optimize(X, y):
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
        # Create a based model
        rf = RandomForestRegressor()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(X, y)
        return grid_search.best_params_
    #====================================================================================================
    # train RF regressor:
    @staticmethod
    def train_rfr(
        X_train, y_train, max_depth = 20, min_samples_leaf = 3, 
        min_samples_split = 12, n_estimators = 500, n_jobs = -1, random_state = 0
    ):
        # regr = MLPRegressor(random_state = 0, hidden_layer_sizes = (500,), max_iter = 1000)
        regr = RandomForestRegressor(
            max_depth = max_depth, 
            min_samples_leaf = min_samples_leaf, 
            # max_features = 10,
            min_samples_split = min_samples_split,
            n_estimators = n_estimators, 
            n_jobs = n_jobs, 
            random_state = random_state
        )
        regr.fit(X_train, y_train)
        return regr
    #====================================================================================================
    # train XGB regressor:
    @staticmethod
    def train_xgb(X_train, y_train):
        xgbr = xgb.XGBRegressor(objective = "reg:squarederror", random_state = 0)
        xgbr.fit(X_train, y_train)
        return xgbr
    #====================================================================================================
    # test XGB regressor:
    @staticmethod
    def test_xgb(xgbr, X_test, y_test, stat = True):
        predicts = xgbr.predict(X_test)[:, np.newaxis]
        df = pd.DataFrame(np.concatenate([y_test, predicts], axis = 1), columns = ["truth", "estimates"])
        if stat:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()["truth"], df.dropna()["estimates"])
            r2 = r_value**2
            mse = mean_squared_error(predicts, y_test)
            rmse = np.sqrt(mse)
            return df, r2, rmse
        else:
            return df
    #====================================================================================================
    # test RF regressor
    @staticmethod
    def test_rfr(regr, X_test, y_test, stat = True):
        predicts = regr.predict(X_test)[:, np.newaxis]
        df = pd.DataFrame(np.concatenate([y_test, predicts], axis = 1), columns = ["truth", "estimates"])
        if stat:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()["truth"], df.dropna()["estimates"])
            r2 = r_value**2
            mse = mean_squared_error(predicts, y_test)
            rmse = np.sqrt(mse)
            return df, r2, rmse
        else:
            return df
    #====================================================================================================
    # apply RF regressor
    @staticmethod
    def apply_rfr(regr, X_apply, df, flux, bench_flux):
        predicts = regr.predict(X_apply)[:, np.newaxis]
        bench_df = df.loc[X_apply.index, bench_flux]
        #-----------------------------------------------------------
        # create dataframe of true flux and predicts:
        df = pd.DataFrame(
            np.concatenate(
                [df.loc[X_apply.index, flux].values, predicts], 
                axis = 1
            ), 
            index = X_apply.index, 
            columns = ["truth", "estimates"]
        )
        slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()["truth"], df.dropna()["estimates"])
        r2 = r_value**2
        df = pd.concat([df, bench_df], axis = 1)
        return df, r2
    #====================================================================================================