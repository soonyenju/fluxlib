import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def nighttime_partitioning_simple(df, NEE, TA, PAR, PAR_threshold = 10):
    '''
    Note: dataframe (df) index is datetime type
    NEE: column label, net ecossytem exchange umol CO2 m-2 s-1
    TA: column label, air temperature (degC)
    PAR: column label, photosynthetically active radiation (W m-2)
    PAR_threshold: threshold to decide if it is day or night (defaul: 10 W m-2)
    '''
    # Function for temperature dependence of respiration (Arrhenius-type model)
    def respiration_model(T, R_base, E0):
        """Exponential function to model ecosystem respiration"""
        return R_base * np.exp(E0 * (T - np.mean(T)) / (8.314 * (T + 273.15)))
    
    df = df.copy()

    # Define day/night based on PAR (photosynthetically active radiation)
    df["is_night"] = df[PAR] < PAR_threshold  # Night if PAR is very low

    # Extract nighttime NEE (Reco proxy)
    nighttime_df = df[df["is_night"]].copy()

    # Fit respiration model using nighttime data
    popt, _ = opt.curve_fit(respiration_model, nighttime_df[TA], nighttime_df[NEE], p0 = [5, 0.1])

    # Extract estimated parameters
    R_base, E0 = popt

    # Compute ecosystem respiration (Reco) for all timestamps
    df["Reco"] = respiration_model(df[TA], R_base, E0)

    # Compute GPP (GPP = Reco - NEE)
    df["GPP"] = df["Reco"] - df[NEE]

    df.loc[df['GPP'] < 0, 'GPP'] = np.nan
    df.loc[df['Reco'] < 0, 'Reco'] = np.nan

    return df[[NEE, 'GPP', 'Reco']]

def daytime_partitioning_simple(df, NEE, TA, PAR, PAR_threshold = 10):
    '''
    Note: dataframe (df) index is datetime type
    The nighttime run is prerequisite for daytime paritioning.
    NEE: column label, net ecossytem exchange umol CO2 m-2 s-1
    TA: column label, air temperature (degC)
    PAR: column label, photosynthetically active radiation (W m-2)
    PAR_threshold: threshold to decide if it is day or night (defaul: 10 W m-2)
    '''
    # Define Michaelis-Menten-type light response function
    def light_response(PAR, alpha, GPP_max, Reco):
        """Light-response function for GPP estimation."""
        return - (alpha * PAR * GPP_max) / (alpha * PAR + GPP_max) + Reco

    df = df.copy()
    df_night_part = nighttime_partitioning_simple(df, NEE, TA, PAR, PAR_threshold=PAR_threshold)
    df = pd.concat([df, df_night_part[['GPP', 'Reco']]], axis = 1)
    # Filter daytime data
    daytime_df = df[df[PAR] > PAR_threshold].copy()

    # Fit light-response function using daytime NEE data
    popt_day, _ = opt.curve_fit(light_response, daytime_df[PAR], daytime_df[NEE], p0=[0.05, 10, 5])

    # Extract estimated parameters
    alpha_opt, GPP_max_opt, Reco_opt = popt_day

    # Compute daytime Reco from the fitted model
    df.loc[df[PAR] > PAR_threshold, "Reco_day"] = Reco_opt

    # Compute daytime GPP (GPP = Reco - NEE)
    df.loc[df[PAR] > PAR_threshold, "GPP_day"] = df.loc[df[PAR] > PAR_threshold, "Reco_day"] - df.loc[df[PAR] > PAR_threshold, NEE]

    # Compare nighttime- and daytime-based Reco estimates
    df["Reco_combined"] = df["Reco"].copy()  # Start with nighttime estimates
    df.loc[df[PAR] > PAR_threshold, "Reco_combined"] = df.loc[df[PAR] > PAR_threshold, "Reco_day"]  # Use daytime estimates where available

    # Compute final GPP using combined Reco
    df["GPP_combined"] = df["Reco_combined"] - df[NEE]

    df = df.drop(['GPP', 'Reco'], axis = 1)
    df = df.rename(columns = {'GPP_combined': 'GPP', 'Reco_combined': 'Reco'})

    df.loc[df['GPP'] < 0, 'GPP'] = np.nan
    df.loc[df['Reco'] < 0, 'Reco'] = np.nan

    return df[[NEE, 'GPP', 'Reco']]

def nighttime_partitioning(df, NEE, TA, PAR, PAR_threshold=10):
    """
    Nighttime partitioning using the Lloyd & Taylor (1994) respiration model.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index and required columns.
    NEE : str
        Column name for net ecosystem exchange (umol CO2 m-2 s-1).
    TA : str
        Column name for air temperature (degC).
    PAR : str
        Column name for photosynthetically active radiation (W m-2).
    PAR_threshold : float, optional
        Threshold to determine night (default: 10 W m-2).

    Returns:
    -------
    df : pandas.DataFrame
        Updated DataFrame with 'Reco' (ecosystem respiration) and 'GPP' (gross primary production).
    """
    def lloyd(T, Rref, E0):
        """Lloyd & Taylor (1994) temperature-dependent respiration model."""
        Tref = 283.15  # 10°C in Kelvin
        T0 = 227.13  # -46.02°C in Kelvin
        return Rref * np.exp(E0 * (1. / (Tref - T0) - 1. / (T - T0)))

    df = df.copy()
    df["is_night"] = df[PAR] < PAR_threshold  # Nighttime flag

    # Convert temperature to Kelvin for Lloyd function
    df["TA_K"] = df[TA] + 273.15  

    # Filter nighttime data
    nighttime_df = df[df["is_night"]].copy()

    # Fit Lloyd & Taylor model using nighttime NEE
    popt, _ = opt.curve_fit(lloyd, nighttime_df["TA_K"], nighttime_df[NEE], p0=[5, 100])

    # Extract estimated respiration parameters
    Rref, E0 = popt

    # Compute ecosystem respiration (Reco) for all timestamps
    df["Reco"] = lloyd(df["TA_K"], Rref, E0)

    # Compute GPP (GPP = Reco - NEE)
    df["GPP"] = df["Reco"] - df[NEE]

    df.loc[df['GPP'] < 0, 'GPP'] = np.nan
    df.loc[df['Reco'] < 0, 'Reco'] = np.nan

    return df[[NEE, 'GPP', 'Reco']]


def daytime_partitioning(df, NEE, TA, PAR, VPD, PAR_threshold=10):
    """
    Daytime partitioning using the Lasslop et al. (2010) light-response model.

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with datetime index and required columns.
    NEE : str
        Column name for net ecosystem exchange (umol CO2 m-2 s-1).
    TA : str
        Column name for air temperature (degC).
    PAR : str
        Column name for photosynthetically active radiation (W m-2).
    VPD : str
        Column name for vapor pressure deficit (hPa).
    PAR_threshold : float, optional
        Threshold to determine daytime (default: 10 W m-2).

    Returns:
    -------
    df : pandas.DataFrame
        Updated DataFrame with refined 'GPP' and 'Reco' estimates.
    """
    def lasslop(Rg, et, VPD, alpha, beta0, k, Rref):
        """
        Lasslop et al. (2010) light-response model with respiration from Lloyd & Taylor (1994).

        Parameters:
        ----------
        Rg : float
            Global radiation (PAR) [W m-2].
        et : float
            Exponential in Lloyd & Taylor model.
        VPD : float
            Vapor Pressure Deficit [Pa].
        alpha : float
            Light use efficiency (initial slope) [umol(C) J-1].
        beta0 : float
            Maximum CO2 uptake rate at VPD0=10 hPa [umol(C) m-2 s-1].
        k : float
            Exponential decrease of max CO2 uptake with VPD [Pa-1].
        Rref : float
            Respiration at Tref=10 degC [umol(C) m-2 s-1].

        Returns:
        -------
        NEE : float
            Net ecosystem exchange (umol CO2 m-2 s-1).
        """
        # Compute Lloyd & Taylor respiration term
        gamma = Rref * et  
        # Vapor pressure deficit limitation on maximum CO2 uptake
        VPD0 = 1000.  # 10 hPa
        kk = np.clip(-k * (VPD - VPD0), -600., 600.)
        beta = np.where(VPD > VPD0, beta0 * np.exp(kk), beta0)
        return -alpha * beta * Rg / (alpha * Rg + beta) + gamma

    # First, apply nighttime partitioning to get initial Reco estimates
    df = df.copy()
    df_night_part = nighttime_partitioning(df, NEE, TA, PAR, PAR_threshold=PAR_threshold)
    df = pd.concat([df, df_night_part[['GPP', 'Reco']]], axis = 1)

    # Convert temperature to Kelvin for Lloyd & Taylor
    df["TA_K"] = df[TA] + 273.15  
    df["et"] = np.exp(100 * (1. / (283.15 - 227.13) - 1. / (df["TA_K"] - 227.13)))

    # Covert hPa to Pa for Lasslop
    df["VPD_hPa"] = df[VPD]
    df[VPD] = df[VPD] * 100.

    # Filter daytime data
    daytime_df = df[df[PAR] > PAR_threshold].copy()

    # Fit Lasslop light-response function
    popt_day, _ = opt.curve_fit(
        lambda Rg, alpha, beta0, k, Rref: lasslop(Rg, daytime_df["et"], daytime_df[VPD], alpha, beta0, k, Rref),
        daytime_df[PAR], daytime_df[NEE], p0=[0.05, 10, 0.005, 5]
    )

    # Extract estimated parameters
    alpha_opt, beta0_opt, k_opt, Rref_opt = popt_day

    # Compute daytime Reco from the fitted model
    df.loc[df[PAR] > PAR_threshold, "Reco_day"] = Rref_opt * df.loc[df[PAR] > PAR_threshold, "et"]

    # Compute daytime GPP using Lasslop model
    df.loc[df[PAR] > PAR_threshold, "GPP_day"] = (
        df.loc[df[PAR] > PAR_threshold, "Reco_day"] - df.loc[df[PAR] > PAR_threshold, NEE]
    )

    # Combine Reco estimates (nighttime for night, daytime for day)
    df["Reco_combined"] = df["Reco"].copy()
    df.loc[df[PAR] > PAR_threshold, "Reco_combined"] = df.loc[df[PAR] > PAR_threshold, "Reco_day"]

    # Compute final GPP using combined Reco
    df["GPP_combined"] = df["Reco_combined"] - df[NEE]

    # Drop intermediate columns and rename final outputs
    df = df.drop(["GPP", "Reco"], axis=1)
    df = df.rename(columns={"GPP_combined": "GPP", "Reco_combined": "Reco"})

    df.loc[df['GPP'] < 0, 'GPP'] = np.nan
    df.loc[df['Reco'] < 0, 'Reco'] = np.nan

    return df[[NEE, 'GPP', 'Reco']]