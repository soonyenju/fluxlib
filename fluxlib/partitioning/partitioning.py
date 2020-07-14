import numpy as np

def neepart_wrapper(func_part, df, undef = -9999, method = "global"):
    df["Tair"] += 273.15
    df["VPD"] *= 100
    df = df.rename(
        columns = {
            'NEE': 'FC', 'Rg': 'SW_IN', 'Tair': 'TA', 'VPD': 'VPD'
        }
    )
    df = df.loc[:, ['FC', 'SW_IN', 'TA', 'VPD']]
    df = df.replace(np.nan, undef)
    dates = df.index
    df = df.values.T
    flag = np.where(df == undef, 2, 0)
    colhead = ['FC', 'SW_IN', 'TA', 'VPD']
    GPP, Reco = func_part(df, flag = flag, date = dates, colhead = colhead, undef = undef, method = method)
    return GPP, Reco