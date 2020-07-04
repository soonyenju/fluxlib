import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

class AuxFiller():
    def __init__(self, pd_series, reverse_date = 10, seed = 42):
        self.pd_series = pd_series
        self.reverse_length = np.int(reverse_date * 48)
        self.seed = seed
    
    def __call__(self):
        self.create_dataset()
        self.train()
        self.test()
        self.apply()

    def create_dataset(self):
        column = self.pd_series.copy()
        for count in range(self.reverse_length):
            move = self.pd_series.values[0: -(count + 1), :]
            pad = np.ones([(count + 1), 1]) * np.nan
            column[str(count).zfill(3)] = np.vstack([pad, move])
        dataset = column.dropna().values
        X = dataset[:, 1::]
        y = dataset[:, 0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state = self.seed)

    def train(self):
        self.xgb_model = xgb.XGBRegressor(objective = "reg:squarederror", random_state = self.seed)
        self.xgb_model.fit(self.X_train, self.y_train)
        with open("aux_fill_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)

    def test(self, savefile = "aux_fill_test"):
        y_pred = self.xgb_model.predict(self.X_test)
        df = pd.DataFrame(
            np.transpose([self.y_test, y_pred]),
            columns = ["truth", "pred"]
        )
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()["truth"], df.dropna()["pred"])
        r2 = r_value**2

        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        df.to_csv(savefile + f"_r2_{np.round(r2, 2)}_rmse_{np.round(rmse, 2)}.csv")

    def apply(self, model_source = None, savefile = "aux_fill_apply.csv"):
        # load model:
        if model_source:
            with open(model_source, "rb") as f:
                xgb_model = pickle.load(f)
        else:
            xgb_model = self.xgb_model
        # fill the gaps:
        df = self.pd_series.copy()
        df = df.interpolate(method = "linear", limit = 2)
        maxv = df.max().values[0]
        minv = df.min().values[0]

        gap_idxs = np.where(np.isfinite(df.values.ravel()) == False)[0]
        for idx in gap_idxs:
            # print(idx)
            if idx < self.reverse_length:
                X = df.iloc[0: idx, :].values.ravel()
                reps = np.ceil(self.reverse_length / idx).astype(np.int)
                X = np.tile(X, reps)[0: self.reverse_length].reshape(1, -1)
            else:
                X = df.iloc[idx - self.reverse_length: idx, :].values.reshape(1, -1)
            pred_val = xgb_model.predict(X)[0]
            # print(df.iloc[idx, :])
            if (pred_val > maxv) or (pred_val < minv):
                if idx == len(df) - 1:
                    pred_val = df.interpolate().values[-1, 0]
                else:
                    pred_val = df.iloc[0: idx + 1, :].interpolate().values[-1, 0]
            df.iloc[idx, :] = pred_val
        df.to_csv(savefile)
            