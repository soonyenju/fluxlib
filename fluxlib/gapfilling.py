import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def sample_continuous_subsets(df, subset_plan, time_col = None, min_valid_ratio = 0.66, max_attempts = 50):
    """
    Randomly selects continuous, non-overlapping subsets from a time series DataFrame,
    ensuring each subset:
        - Has a specific length as defined in `subset_plan`.
        - Appears a specific number of times, avoiding overlaps.
        - Has at least `min_valid_ratio` valid (non-NaN) values.

    Args:
        df (pd.DataFrame): Input time series DataFrame.
        subset_plan (dict): Dictionary {subset_length: count}.
        time_col (str): Optional column name to sort by if time-based ordering is required.
        min_valid_ratio (float): Minimum fraction of non-NaN values required in a subset.
        max_attempts (int): Maximum retries to find a valid subset.

    Returns:
        list of pd.DataFrame: A list of sampled, non-overlapping DataFrame subsets.
    """
    if time_col:
        df = df.sort_values(by=time_col).reset_index(drop=True)  # Ensure time-ordering
    else:
        df = df.sort_index()  # Sort by index if no time column is provided

    subsets = []
    selected_ranges = []  # Track used (start_idx, end_idx) ranges to prevent overlaps
    max_index = len(df) - 1

    for length, count in subset_plan.items():
        for _ in range(count):
            valid_subset_found = False
            attempts = 0

            while not valid_subset_found and attempts < max_attempts:
                if max_index - length + 1 <= 0:  # Ensure there's space for selection
                    break

                start_idx = np.random.randint(0, max_index - length + 1)
                end_idx = start_idx + length - 1

                # Ensure no overlap
                if any(start <= end_idx and end >= start_idx for start, end in selected_ranges):
                    attempts += 1
                    continue  # Retry with a new selection

                subset = df.iloc[start_idx : start_idx + length]  # Extract subset

                # Check valid (non-NaN) ratio
                valid_ratio = subset.notna().mean().mean()
                if valid_ratio >= min_valid_ratio:
                    subsets.append(subset)
                    selected_ranges.append((start_idx, end_idx))  # Mark range as used
                    valid_subset_found = True

                attempts += 1

    return subsets
'''
# Example Usage:

data = {
    "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "value": np.random.randn(100)
}

# Introduce NaNs randomly
nan_indices = np.random.choice(100, size=30, replace=False)  # Randomly pick 30 indices
data["value"][nan_indices] = np.nan

df = pd.DataFrame(data)

# Define specific subset lengths and counts
subset_plan = {5: 4, 10: 2, 15: 1}  # 4 subsets of length 5, 2 of 10, and 1 of 15
sampled_subsets = sample_continuous_subsets(df, subset_plan, time_col="timestamp")

# Print results
for i, subset in enumerate(sampled_subsets):
    print(f"Subset {i+1} (length={len(subset)}):\n", subset, "\n")
'''

def gap_filling_training(X_train, y_train, n_models = 100, frac_sample = 0.8):
    base_params_xgb = {
        "objective": "reg:squarederror",
        'seed': 0,
        "random_state": 0,
    }
    params_xgb = deepcopy(base_params_xgb)
    # dropout-like regularization
    params_xgb.update({
        "subsample": 0.8,  # Use 80% of the data for each tree
        "colsample_bytree": 0.8,  # Use 80% of the features for each tree
    })

    models = []
    for i in tqdm(range(n_models)):
        # Create a bootstrapped dataset
        y_resampled = y_train.copy().sample(frac = frac_sample, random_state = i)
        X_resampled = X_train.copy().loc[y_resampled.index]
        # print(y_resampled.sort_index().index[0], y_resampled.sort_index().index[-1])

        # Train the XGBoost model
        params_xgb.update({'random_state': i})
        model = XGBRegressor(**params_xgb)
        model.fit(X_resampled, y_resampled)
        models.append(model)
    return models

def get_metrics(df, truth = 'truth', pred = 'pred', return_dict = False):
    '''
    Calculate statistical measures between validation and prediction sequences
    '''
    df = df[[truth, pred]].copy().dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df.dropna()[truth], df.dropna()[pred])
    r2 = r_value**2
    mse = mean_squared_error(df.dropna()[truth], df.dropna()[pred])
    rmse = np.sqrt(mse)
    mbe = np.mean(df.dropna()[pred] - df.dropna()[truth])
    mae = (df.dropna()[pred] - df.dropna()[truth]).abs().mean()
    if return_dict:
        return pd.DataFrame.from_dict([{
            'R2': r2, 
            'Slope': slope, 
            'RMSE': rmse, 
            'MBE': mbe, 
            'MAE': mae, 
            'Intercept': intercept, 
            'p-value': p_value, 
            'std_err': std_err
        }])
    else:
        return r2, slope, rmse, mbe, mae, intercept, p_value, std_err

def gap_filling_validation(models, art_gaps, Xs, X_names):
    y_test_all = []
    y_pred_all = []
    for gap in art_gaps:
        y_pred = []
        for i in range(len(models)):
            model = models[i]
            X_test = Xs.loc[gap.index, X_names]
            y_pred_t = model.predict(X_test)
            y_pred_t = pd.DataFrame(y_pred_t, index = gap.index, columns = ['pred_' + str(i).zfill(2)])
            y_pred.append(y_pred_t)
        y_pred = pd.concat(y_pred, axis = 1)
        y_pred_all.append(y_pred)
        y_test_all.append(gap)

    y_pred_all = pd.concat(y_pred_all, axis = 0)
    y_test_all = pd.concat(y_test_all, axis = 0)

    dfo = pd.concat([y_test_all, y_pred_all.mean(axis = 1).rename('pred')], axis = 1)
    eval_res = get_metrics(dfo, truth = 'truth', pred = 'pred', return_dict = True)
    eval_res['MEAN'] = dfo['truth'].mean()
    # eval_res['UNC'] = y_pred_all.std(axis = 1).mean()
    return eval_res

'''
Example for loading multiple files and filling gaps:

df_eval_res = []
paths = list(root_proj.joinpath('Flux_data').glob('*.csv'))
for cnt, p in enumerate(paths):
    site = p.stem.split('_DATA')[0]
    # if site != 'ZM_MON': continue
    print(f"............{cnt + 1} of {len(paths)}: {site}............")
    df_ec = pd.read_csv(p, encoding='unicode_escape', index_col = 0)
    df_ec.index = pd.to_datetime(df_ec.index, format = '%Y-%m-%d %H:%M:%S')

    if 'c_co2_flux_µmol.m-2.s-1' in df_ec.columns: df_ec = df_ec.rename(columns = {'c_co2_flux_µmol.m-2.s-1': 'co2_flux_µmol.m-2.s-1'})
    cflux_name = 'co2_flux_µmol.m-2.s-1'
    if cflux_name in df_ec.columns:
        mask = (df_ec[cflux_name] > 100) | (df_ec[cflux_name] < -100)
        df_ec.loc[mask, cflux_name] = np.nan
    hflux_name = 'latent_heat_W/m²'
    if hflux_name in df_ec.columns:
        mask = df_ec[hflux_name] < 0
        df_ec.loc[mask, hflux_name] = np.nan
    # --------------------------------------------------------------------------
    df_ec['month'] = df_ec.index.month
    df_ec['day'] = df_ec.index.day
    df_ec['hour'] = df_ec.index.hour

    # ==========================================================================

    if not cflux_name in df_ec.columns: continue
    X_names = ['ERA5_Temp_C', 'ERA5_SSR_W.M2', 'ERA5_LWR_W.M2', 'ERA5_Wind_Speed_m.s', 'ERA5_Precip_mm', 'VPD_hPa', 'month', 'day', 'hour']
    y_name = ['co2_flux_µmol.m-2.s-1']

    subset_plan = {48: 20, 48 * 7: 10, 48 * 30: 1} # short gaps (24 h, 20% of total gaps), long gaps (7-day, 30% of total gaps) and very-long gaps (30-day, 50% of total gaps).

    Xs = df_ec[X_names]
    ys = df_ec[y_name]
    art_gaps = sample_continuous_subsets(ys, subset_plan)

    index_train = ys.dropna().index
    for gap in art_gaps:
        gap.columns = ['truth']
        index_train = index_train.difference(gap.index)

    X_train = Xs.loc[index_train, X_names]
    y_train = ys.loc[index_train, y_name]
    #-------------------------------------------------------------------------------
    models = gap_filling_training(X_train, y_train, n_models = 10, frac_sample = 0.1)
    eval_res = gap_filling_validation(models, art_gaps, Xs, X_names)
    eval_res.index = [site]
    # ------------------------------------------------------------------------------
    index_real_gaps = ys.index#.difference(ys.dropna().index)
    preds = []
    for i in range(len(models)):
        model = models[i]
        pred = model.predict(Xs.loc[index_real_gaps, X_names])
        pred = pd.DataFrame(pred, index = index_real_gaps, columns = ['pred_' + str(i).zfill(2)])
        preds.append(pred)
    preds = pd.concat(preds, axis = 1)

    dfo = pd.concat([ys, preds], axis = 1)

    df_eval_res.append(eval_res)
    # google.download_file(dfo, f'{site}_gapfilling.csv')
    dfo.to_csv(root_proj.joinpath('output').joinpath(f'{site}_gapfilling.csv'))
df_eval_res = pd.concat(df_eval_res, axis = 0)
df_eval_res.to_csv(root_proj.joinpath('output').joinpath(f'validation_metrics.csv'))

'''