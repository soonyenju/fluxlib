import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

class AuxFiller():
    def __init__(self, df, varname, scale = "15T", seed = 0):
        self.varname = varname
        self.seed = seed
        self.dfv = self.preproc(df, self.varname, scale)
    def __call__(self, validation = 0):
        return self.fill(self.dfv, self.varname, validation = validation, seed = self.seed)
        
    def preproc(self, df, varname, scale):
        dts = df.index
        # dfv = np.cos(np.radians(df[[varname]].copy()))
        dfv = df[[varname]].copy()
        # print(dfv)
        # -------------------------------------------------------
        ds = dfv.resample("D").mean()
        dfv["mean"] = ds.resample(scale).bfill()
        # dfv["mean"] = df["mean"]#.ffill()
        # -------------------------------------------------------

        hours = dts.hour
        days = np.array([d.days for d in dts.date - dts.date[0]])
        dfv["hours"] = hours
        dfv["days"] = days
        dfv["doys"] = dts.dayofyear
        dfv["seasons"] = (dts.month%12 + 3) // 3 
        return dfv

    def fill(self, dfv, varname, drivers = ["mean", "hours", "days", "doys", "seasons"], validation = 0, seed = 0):
        if validation:
            np.random.seed(seed)
            Xs = dfv.dropna()[drivers].values
            ys = dfv.dropna()[varname].values
            sss = ShuffleSplit(n_splits=1, test_size = 0.33, random_state = seed)
            sss.get_n_splits(Xs, ys)
            train_index, test_index = next(sss.split(Xs, ys)) 

            X_train, X_test = Xs[train_index], Xs[test_index] 
            y_train, y_test = ys[train_index], ys[test_index]
            # --------------------------------------------------------------------------------------
            xgb_model = xgb.XGBRegressor(objective = "reg:squarederror", random_state = seed)
            xgb_model.fit(X_train, y_train)

            y_pred = xgb_model.predict(X_test)
            dft = pd.DataFrame(
                np.transpose([y_test, y_pred]),
                columns = ["truth", "pred"]
            )
            
            dft = dft.dropna()

            slope, intercept, r_value, p_value, std_err = stats.linregress(dft["truth"], dft["pred"])
            r2 = r_value**2

            mse = mean_squared_error(dft["truth"], dft["pred"])
            rmse = np.sqrt(mse)
            
            bias = np.mean(dft["pred"] - dft["truth"])
            
            report = {
                "R2": np.round(r2, 2),
                "Slope": np.round(slope, 2),
                "RMSE": np.round(rmse, 2),
                "Bias": np.round(bias, 2)
            }
            return report
            # --------------------------------------------------------------------------------------
        else:
            np.random.seed(seed)
            empty_index = dfv.index[dfv[varname].isna()]
            valid_index = dfv.index[~dfv[varname].isna()]
            X_train = dfv.loc[valid_index, drivers].interpolate().bfill()
            y_train = dfv.loc[valid_index, varname]
            X_app = dfv.loc[empty_index, drivers].interpolate().bfill() 
            xgb_model = xgb.XGBRegressor(objective = "reg:squarederror", random_state = seed)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_app)
            dfv.loc[empty_index, varname] = y_pred
            return dfv[[varname]]
    
# example        
# auxfiller = AuxFiller(df, "VPD")
# auxfiller(validation = 1)