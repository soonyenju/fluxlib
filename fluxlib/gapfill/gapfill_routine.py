import yaml
from scitbx import Yaml

def create_gapfill(df, flux, cfg_name, savefile = 'ggapfill.yaml'):
    if cfg_name == 'RFR':
        cfg = {
            'regressor': 'RFR',
            'params':
                {
                    'max_depth': 20,
                    'min_samples_leaf': 3,
                    'min_samples_split': 12,
                    'n_estimators': 100,
                    'n_jobs': -1
                }
        }
    elif cfg_name == 'XGB':
        cfg = {
            'regressor': 'XGB',
            'params': 
                {
                    'objective': 'reg:squarederror',
                    'random_state': 0
                }
        }
    elif cfg_name == 'SVR':
        cfg = {
            'regressor': 'SVR',
            'params':
                {
                    'kernel': 'rbf'
                }
        }
    elif cfg_name == 'MLP':
        cfg = {
            'regressor': 'MLP',
            'params':
                {
                    'learning_rate': 'adaptive' 
                }           
        }
    elif cfg_name == 'GBR':
        cfg = {
            'regressor': 'GBR',
            'params':
                {
                    'loss': 'ls'
                }
        }
    elif cfg_name == 'ABR':
        cfg = {
            'regressor': 'ABR',
            'params':
                {
                    'random_state': 0
                }
            
        }
    else:
        raise Exception('Wrong machine-learning name')
    # --------------------------------------------------------------------------    
    cfg.update({
        'drivers': df.columns.drop(flux).tolist()
    })
    cfg.update({
        'flux': [flux]
    })
    cfg.update({
        'rg': ['PPFD']
    })

    with open(savefile, 'w') as file:
        documents = yaml.dump(cfg, file, sort_keys=False)

# ===========================================================================================================================================================
def artificial_gaps(scenario, savefile = 'make_gaps_cfg.yaml'):
    if scenario == 'very-short':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 1,
                    'ratio': 1.0
                    }
                ],
            'tag': "very_short"
        }
    elif scenario == 'short':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 8,
                    'ratio': 1.0
                    }
                ],
            'tag': "short"
        }
    elif scenario == 'medium':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 64,
                    'ratio': 1.0
                    }
                ],
            'tag': "medium"
        }
    elif scenario == 'long':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 576,
                    'ratio': 1.0
                    }
                ],
            'tag': "long"
        }
    elif scenario == 'mixed':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 1,
                    'ratio': 0.25
                    },
                    {
                    'window_size': 8,
                    'ratio': 0.25
                    },
                    {
                    'window_size': 64,
                    'ratio': 0.25
                    },
                    {
                    'window_size': 576,
                    'ratio': 0.25
                    },
                ],
            'tag': "mixed"
        }
    elif scenario == 'very-long':
        cfg = {
            'tgr': 0.25,
            'gaps':
                [
                    {
                    'window_size': 1440, # 48 * 30,
                    'ratio': 0.5
                    },
                    {
                    'window_size': 336,
                    'ratio': 0.3
                    },
                    {
                    'window_size': 48,
                    'ratio': 0.2
                    }
                ],
            'tag': "very_long"
        }
    elif scenario == '1mon':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 1440,
                    'ratio': 1.0
                    }
                ],
            'tag': "1mon"
        }
    elif scenario == '2mon':
        cfg = {
            'tgr': 0.1,
            'gaps':
                [
                    {
                    'window_size': 2880,
                    'ratio': 1.0
                    }
                ],
            'tag': "2mon"
        }
    elif scenario == '3mon':
        cfg = {
            'tgr': 0.2,
            'gaps':
                [
                    {
                    'window_size': 4320,
                    'ratio': 1.0
                    }
                ],
            'tag': "3mon"
        }
    else:
        raise Exception('Wrong scenario')
        
        
    with open(savefile, 'w') as file:
        documents = yaml.dump(cfg, file, sort_keys=False)

def gapfill_pipeline(df, flux, scenario, rbak = 10, vtheta = 0.1, config_path = "make_gaps_cfg.yaml", ggapfill_cfg = "ggapfill.yaml"):
    import numpy as np
    import pandas as pd
    from . import utils
    from .ggapfill import GFiller
    print(Yaml(config_path).load())
    
    if scenario == 'last-one-third':
        valid_idx = np.where(np.isfinite(df[flux]))[0]
        df_ = df[flux].dropna()
        anchor_t = df_.index[len(df_) // 3 * (3 - 1)]
        anchor = df.index.get_loc(anchor_t)
        train_idx = np.intersect1d(valid_idx, np.arange(0, anchor))
        test_idx = np.intersect1d(valid_idx, np.arange(anchor, len(df)))
        tags = np.array(['last-one-third']).repeat(len(test_idx))
    else:
        train_idx, test_idx, tags = utils.make_gap_pipeline(config_path, df, flux, rbak = rbak, vtheta = vtheta)

    filler = GFiller(ggapfill_cfg)

    res, app = filler.run_filling_pipeline(df, itrain = train_idx, itest = test_idx)
    res['tags'] = tags
    return res, app, filler

class GapfillReport:
    def __init__(self):
      pass

    @staticmethod
    def get_filled_report(df):
        df = df.copy()
        for flux in [c for c in df.columns if not c.endswith('filled')]:
            flag = flux + '_flag'
            df[flag] = 1
            df.loc[df[flux].dropna().index, flag] = 0

        df = df[[c for c in df.columns if c.endswith('filled')] + [c for c in df.columns if c.endswith('flag')]]
        df.columns = [c.replace('_filled', '') for c in df.columns]
        return df

    @staticmethod
    def get_metrics_report(df):

        def get_metrics(x, y):
            import numpy as np
            from scipy import stats
            from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
            mse = mean_squared_error(x, y)
            r2 = rvalue ** 2
            rmse = np.sqrt(mse)
            mbe = (y - x).mean()
            mae = mean_absolute_error(x, y)
            # evs = explained_variance_score(x, y)
            return {
                "R2": r2,
                "SLOPE": slope,
                "RMSE": rmse,
                "MBE": mbe,
                'MAE': mae,
                "INTERCEPT": intercept,
                "PVALUE": pvalue,
                "STDERR": stderr,
                # "EXPLAINED_VARIANCE": evs,
                "MAE": mae,
            }

        df = df.copy()
        dictm = {}
        for flux in [c.replace('truth_', '') for c in df.columns if c.startswith('truth')]:
            dft = df[['truth_' + flux, 'estimates_' + flux]].dropna()
            x = dft['truth_' + flux]
            y = dft['estimates_' + flux]
            metrics = get_metrics(x, y)
            # metrics = roundit(metrics)
            # dfm = pd.DataFrame(metrics.items())
            dictm[flux] = metrics
        return dictm

    @staticmethod
    def get_filler_report(filler_dict):
        import pandas as pd

        def get_filler_info(filler):
            df_imp = pd.DataFrame(filler.impurity_feature_importance(filler.regr), index = ['importance_mean', 'importance_std'], columns = filler.feature_names).T # permutation_feature_importance
            cfg = filler.cfg
            return cfg, df_imp

        df_imps = []
        for flux, filler in filler_dict.items():
            cfg, df_imp = get_filler_info(filler)
            df_imp.columns = [flux + '_' + c for c in df_imp.columns]
            df_imps.append(df_imp)
        df_imps = pd.concat(df_imps, axis = 1)
        return cfg, df_imps

    @staticmethod
    def get_model_report(filler_dict):
        regr_dict = {}
        for flux, filler in filler_dict.items():
            regr_dict[flux] = filler
        return regr_dict

    @staticmethod
    def get_artificial_gap_report(config_path = "make_gaps_cfg.yaml"):
        gap_cfg = Yaml(config_path).load()
        return gap_cfg

    def get_full_report(self, app, res, regr, config_path = "make_gaps_cfg.yaml"):
        df_filled = GapfillReport.get_filled_report(app)
        dict_metrics = GapfillReport.get_metrics_report(res)
        cfg_filler, df_importance = GapfillReport.get_filler_report(regr)
        dict_models = GapfillReport.get_model_report(regr)
        cfg_art_gap = GapfillReport.get_artificial_gap_report(config_path = config_path)

        report = {
            'FLUXES': df_filled,
            'METRICS': dict_metrics,
            'MODELS': dict_models,
            'FEATIMP': df_importance, # feature importance
            'ALGOCFG': cfg_filler, # algorithm configuretion
            'ARTGAPCFG': cfg_art_gap, # artificial gap configuration
        }
        return report