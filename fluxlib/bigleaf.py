"""
Converting bigleaf in r to Python
Credit: https://github.com/cran/bigleaf/tree/master/R
It's still under debugging, use with cautions
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


# =========================================================================================================
# WUE_metrics.r

def WUE_metrics(data, GPP_col="GPP", NEE_col="NEE", LE_col="LE", VPD_col="VPD", Tair_col="Tair",
                constants=None):
    """
    Calculate WUE, WUE_NEE, IWUE, and uWUE metrics from flux data.

    Parameters:
    - data: pandas DataFrame with flux data
    - GPP_col, NEE_col, LE_col, VPD_col, Tair_col: column names in data
    - constants: dictionary of unit conversion constants

    Returns:
    - Dictionary of median WUE metrics
    """
    if constants is None:
        constants = bigleaf_constants()

    # Extract relevant columns
    GPP = data[GPP_col]
    NEE = data[NEE_col]
    LE = data[LE_col]
    VPD = data[VPD_col]
    Tair = data[Tair_col]

    # Convert LE to ET
    ET = LE_to_ET(LE, Tair)

    # Convert GPP and NEE to gC m-2 s-1
    GPP_gC = GPP * constants["umol2mol"] * constants["Cmol"] * constants["kg2g"]
    NEE_gC = NEE * constants["umol2mol"] * constants["Cmol"] * constants["kg2g"]

    # Avoid divide-by-zero or NaN by masking invalid ET values
    valid_mask = (ET > 0) & GPP_gC.notna() & NEE_gC.notna() & VPD.notna()

    # Compute metrics
    WUE = np.median((GPP_gC[valid_mask] / ET[valid_mask]).dropna())
    WUE_NEE = np.median((np.abs(NEE_gC[valid_mask]) / ET[valid_mask]).dropna())
    IWUE = np.median(((GPP_gC[valid_mask] * VPD[valid_mask]) / ET[valid_mask]).dropna())
    uWUE = np.median(((GPP_gC[valid_mask] * np.sqrt(VPD[valid_mask])) / ET[valid_mask]).dropna())

    return {
        "WUE": WUE,
        "WUE_NEE": WUE_NEE,
        "IWUE": IWUE,
        "uWUE": uWUE
    }

# =========================================================================================================
# aerodynamic_conductance.r

def aerodynamic_conductance(data, Tair="Tair", pressure="pressure", wind="wind", ustar="ustar", H="H",
                             zr=None, zh=None, d=None, z0m=None, Dl=None, N=2, fc=None, LAI=None,
                             Cd=0.2, hs=0.01, wind_profile=False, stab_correction=True,
                             stab_formulation="Dyer_1970", Rb_model="Thom_1972",
                             kB_h=None, Sc=None, Sc_name=None, constants=None):

    # Handle default constants
    if constants is None:
        constants = {
            'k': 0.41,           # von Karman constant
            'cp': 1005,          # specific heat of air (J K-1 kg-1)
            'Kelvin': 273.15,    # C to K conversion
            'g': 9.81,           # gravity (m s-2)
            'pressure0': 101.3,  # reference pressure at sea level (kPa)
            'Tair0': 298.15,     # reference temperature (K)
            'Sc_CO2': 0.9,       # Schmidt number for CO2
            'Pr': 0.71           # Prandtl number
        }

    Rb_model = Rb_model if Rb_model in ["Thom_1972", "Choudhury_1988", "Su_2001", "constant_kB-1"] else "Thom_1972"
    stab_formulation = stab_formulation if stab_formulation in ["Dyer_1970", "Businger_1971"] else "Dyer_1970"

    # Check for required columns in data
    for col in [Tair, pressure, wind, ustar, H]:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in input data.")

    # Calculate canopy boundary layer conductance (Gb)
    if Rb_model in ["Thom_1972", "Choudhury_1988", "Su_2001"]:
        if Rb_model == "Thom_1972":
            Gb_mod = gb_thom(ustar=data[ustar], Sc=Sc, Sc_name=Sc_name, constants=constants)

        elif Rb_model == "Choudhury_1988":
            Gb_mod = gb_choudhury(data, Tair=Tair, pressure=pressure, wind=wind, ustar=ustar, H=H,
                                  leafwidth=Dl, LAI=LAI, zh=zh, zr=zr, d=d, z0m=z0m,
                                  stab_formulation=stab_formulation, Sc=Sc, Sc_name=Sc_name,
                                  constants=constants)

        elif Rb_model == "Su_2001":
            Gb_mod = gb_su(data, Tair=Tair, pressure=pressure, wind=wind, ustar=ustar, H=H,
                           zh=zh, zr=zr, d=d, z0m=z0m, Dl=Dl, N=N, fc=fc, LAI=LAI,
                           Cd=Cd, hs=hs, stab_formulation=stab_formulation,
                           Sc=Sc, Sc_name=Sc_name, constants=constants)

        kB_h = Gb_mod["kB_h"]
        Rb_h = Gb_mod["Rb_h"]
        Gb_h = Gb_mod["Gb_h"]
        Gb_x = Gb_mod[[col for col in Gb_mod.columns if col.startswith("Gb_") and col != "Gb_h"]]

    elif Rb_model == "constant_kB-1":
        if kB_h is None:
            raise ValueError("kB_h must be specified when Rb_model is 'constant_kB-1'")
        Rb_h = kB_h / (constants["k"] * data[ustar])
        Gb_h = 1.0 / Rb_h
        Gb_x = pd.DataFrame()  # Placeholder for additional quantities

    # Additional aerodynamic resistance and conductance calculations would go here...

    return {
        "kB_h": kB_h,
        "Rb_h": Rb_h,
        "Gb_h": Gb_h,
        **Gb_x.to_dict(orient="series")
    }

# =========================================================================================================
# bigleaf_constants.r

def bigleaf_constants(**kwargs):
    """
    Global constants used in the bigleaf model.

    You can override any constant by passing it as a keyword argument.
    For example: bigleaf_constants(k=0.4)
    """

    constants = {
        # Physical constants
        'cp': 1004.834,           # specific heat of air for constant pressure (J K-1 kg-1)
        'Rgas': 8.31451,          # universal gas constant (J mol-1 K-1)
        'Rv': 461.5,              # gas constant of water vapor (J kg-1 K-1)
        'Rd': 287.0586,           # gas constant of dry air (J kg-1 K-1)
        'Md': 0.0289645,          # molar mass of dry air (kg mol-1)
        'Mw': 0.0180153,          # molar mass of water vapor (kg mol-1)
        'eps': 0.622,             # Mw / Md
        'g': 9.81,                # gravitational acceleration (m s-2)
        'solar_constant': 1366.1,# solar radiation at earth (W m-2)
        'pressure0': 101325,      # reference atmospheric pressure at sea level (Pa)
        'Tair0': 273.15,          # reference air temperature (K)
        'k': 0.41,                # von Karman constant
        'Cmol': 0.012011,         # molar mass of carbon (kg mol-1)
        'Omol': 0.0159994,        # molar mass of oxygen (kg mol-1)
        'H2Omol': 0.01801528,     # molar mass of water (kg mol-1)
        'sigma': 5.670367e-08,    # Stefan-Boltzmann constant (W m-2 K-4)
        'Pr': 0.71,               # Prandtl number
        'Sc_CO2': 1.07,           # Schmidt number for CO2
        'Le067': 0.93,            # Lewis number for water vapor^0.67

        # Conversion constants
        'Kelvin': 273.15,         # Celsius to Kelvin
        'DwDc': 1.6,              # Diffusivity ratio water vapor/CO2
        'days2seconds': 86400,    # seconds per day
        'kPa2Pa': 1000,           # kilopascal to pascal
        'Pa2kPa': 0.001,          # pascal to kilopascal
        'umol2mol': 1e-06,        # micromole to mole
        'mol2umol': 1e06,         # mole to micromole
        'kg2g': 1000,             # kilogram to gram
        'g2kg': 0.001,            # gram to kilogram
        'kJ2J': 1000,             # kilojoule to joule
        'J2kJ': 0.001,            # joule to kilojoule
        'se_median': 1.253,       # SE mean to SE median
        'frac2percent': 100       # fraction to percent
    }

    # Override defaults if any keyword arguments are provided
    constants.update(kwargs)

    return constants

# =========================================================================================================
# bigleaf_physiology.r

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar

# Constants
# def bigleaf_constants():
#     return {
#         'DwDc': 1.6,
#         'Kelvin': 273.15,
#         'Rgas': 8.314,
#         'kJ2J': 1000,
#         'J2kJ': 0.001,
#         'se_median': 1.253
#     }

# 1. Intercellular CO2
def intercellular_CO2(data, Ca='Ca', GPP='GPP', Gs='Gs_mol', Rleaf=None, missing_Rleaf_as_NA=False, constants=None):
    constants = constants or bigleaf_constants()
    Rleaf = 0 if Rleaf is None else data[Rleaf] if missing_Rleaf_as_NA else data[Rleaf].fillna(0)
    Ci = data[Ca] - (data[GPP] - Rleaf) / (data[Gs] / constants['DwDc'])
    return Ci

# 2. Arrhenius Temperature Response
def arrhenius_temp_response(param, Temp, Ha, Hd=None, dS=None, constants=None):
    constants = constants or bigleaf_constants()
    Temp += constants['Kelvin']
    Tref = 25.0 + constants['Kelvin']
    Ha *= constants['kJ2J']
    if Hd is not None: Hd *= constants['kJ2J']
    if dS is not None: dS *= constants['kJ2J']
    if Hd is None or dS is None:
        return param / np.exp(Ha * (Temp - Tref) / (Tref * constants['Rgas'] * Temp))
    term1 = np.exp(Ha * (Temp - Tref) / (Tref * constants['Rgas'] * Temp))
    term2 = (1 + np.exp((Tref * dS - Hd) / (Tref * constants['Rgas'])))
    term3 = (1 + np.exp((Temp * dS - Hd) / (Temp * constants['Rgas'])))
    return param / (term1 * term2 / term3)

# 3. Photosynthetic Capacity
def photosynthetic_capacity(data, Temp, GPP='GPP', Ci=None, PPFD='PPFD', Rleaf=None,
                            Oi=0.21, Kc25=404.9, Ko25=278.4, Gam25=42.75,
                            Kc_Ha=79.43, Ko_Ha=36.38, Gam_Ha=37.83,
                            Vcmax_Ha=65.33, Vcmax_Hd=200, Vcmax_dS=0.635,
                            Jmax_Ha=43.9, Jmax_Hd=200, Jmax_dS=0.640,
                            Theta=0.7, alpha_canopy=0.8, missing_Rleaf_as_NA=False,
                            Ci_C4=100, C3=True, PPFD_j=(200, 500), PPFD_c=1000,
                            constants=None):
    constants = constants or bigleaf_constants()
    TempK = data[Temp] + constants['Kelvin']
    Tref = 25.0 + constants['Kelvin']
    Rleaf = 0 if Rleaf is None else data[Rleaf] if missing_Rleaf_as_NA else data[Rleaf].fillna(0)
    Kc_Ha *= constants['kJ2J']
    Ko_Ha *= constants['kJ2J']
    Gam_Ha *= constants['kJ2J']
    Kc = Kc25 * np.exp(Kc_Ha * (TempK - Tref) / (Tref * constants['Rgas'] * TempK))
    Ko = Ko25 * np.exp(Ko_Ha * (TempK - Tref) / (Tref * constants['Rgas'] * TempK))
    Gam = Gam25 * np.exp(Gam_Ha * (TempK - Tref) / (Tref * constants['Rgas'] * TempK))
    Ko *= constants['J2kJ']
    Ci = Ci.copy()
    Ci[Ci < 80] = np.nan
    GPP_data = data[GPP]
    PPFD_data = data[PPFD]
    GPPc = GPP_data.where(PPFD_data >= PPFD_c)
    GPPj = GPP_data.where((PPFD_data >= PPFD_j[0]) & (PPFD_data <= PPFD_j[1]))
    Vcmax = (GPPc - Rleaf) * (Ci + Kc * (1 + Oi / Ko)) / (Ci - Gam)
    J = (GPPj - Rleaf) * (4 * Ci + 8 * Gam) / (Ci - Gam)
    APPFD_PSII = PPFD_data * alpha_canopy * 0.85 * 0.5
    Jmax = []
    for i in J.dropna().index:
        result = minimize_scalar(
            lambda jmax: abs(J[i] - ((APPFD_PSII[i] + jmax - np.sqrt((APPFD_PSII[i] + jmax)**2 - 4 * Theta * APPFD_PSII[i] * jmax)) / (2 * Theta))),
            bounds=(0, 1000), method='bounded'
        )
        Jmax.append(result.x if result.success else np.nan)
    Jmax = np.array(Jmax)
    Vcmax25 = arrhenius_temp_response(Vcmax, TempK - constants['Kelvin'], Vcmax_Ha, Vcmax_Hd, Vcmax_dS, constants)
    Jmax25 = arrhenius_temp_response(Jmax, TempK.loc[J.dropna().index] - constants['Kelvin'], Jmax_Ha, Jmax_Hd, Jmax_dS, constants)
    return {
        "Vcmax25": round(np.nanmedian(Vcmax25), 2),
        "Vcmax25_SE": round(constants['se_median'] * np.nanstd(Vcmax25) / np.sqrt(np.sum(~np.isnan(Vcmax25))), 2),
        "Jmax25": round(np.nanmedian(Jmax25), 2),
        "Jmax25_SE": round(constants['se_median'] * np.nanstd(Jmax25) / np.sqrt(np.sum(~np.isnan(Jmax25))), 2)
    }

# 4. Stomatal Slope (Medlyn model)
def stomatal_slope(data, GPP='GPP', Ca='Ca', VPD='VPD', Ci=None, constants=None):
    constants = constants or bigleaf_constants()
    Ci = data[Ci] if isinstance(Ci, str) else Ci
    VPD = data[VPD]
    Ca = data[Ca]
    GPP = data[GPP]
    m = GPP * Ca / (Ci - constants['DwDc'] * Ci * VPD)
    return m

# 5. Light Response (Rectangular hyperbola)
def light_response(PPFD, alpha, Amax, theta):
    return ((alpha * PPFD + Amax) - np.sqrt((alpha * PPFD + Amax)**2 - 4 * alpha * PPFD * Amax * theta)) / (2 * theta)

# 6. Light Use Efficiency
def light_use_efficiency(data, PPFD='PPFD', GPP='GPP', model=False, start_vals=(0.01, 20, 0.85), bounds=None):
    PPFD = data[PPFD].values
    GPP = data[GPP].values
    LUE = GPP / PPFD
    if not model:
        return np.nanmedian(LUE)
    bounds = bounds or ([0, 0, 0], [np.inf, np.inf, 1])
    popt, _ = curve_fit(light_response, PPFD, GPP, p0=start_vals, bounds=bounds)
    return popt[0]  # alpha

# 7. Stomatal Sensitivity (Slope of ln(GPP) vs ln(VPD))
def stomatal_sensitivity(data, GPP='GPP', VPD='VPD', constants=None):
    constants = constants or bigleaf_constants()
    ln_GPP = np.log(data[GPP])
    ln_VPD = np.log(data[VPD])
    mask = (~np.isnan(ln_GPP)) & (~np.isnan(ln_VPD)) & (ln_VPD != -np.inf)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(ln_VPD[mask], ln_GPP[mask], 1)
    return slope

# =========================================================================================================
# boundary_layer_conductance.r

import numpy as np
import pandas as pd

def gb_thom(ustar, Sc=None, Sc_name=None, constants=None):
    Rb_h = 6.2 * ustar ** -0.667
    Gb_h = 1 / Rb_h
    kB_h = Rb_h * constants['k'] * ustar

    if Sc is not None or Sc_name is not None:
        if len(Sc) != len(Sc_name):
            raise ValueError("Arguments 'Sc' and 'Sc_name' must have the same length")
        if not isinstance(Sc, (list, np.ndarray)):
            raise TypeError("Argument 'Sc' must be numeric")

    Sc_full = [constants['Sc_CO2']] + (Sc if Sc else [])
    Gb_x = {f"Gb_{name}": Gb_h / ((s / constants['Pr']) ** 0.67)
            for s, name in zip(Sc_full, ['CO2'] + (Sc_name if Sc_name else []))}

    return pd.DataFrame({'Gb_h': Gb_h, 'Rb_h': Rb_h, 'kB_h': kB_h, **Gb_x})


def gb_choudhury(data, leafwidth, LAI, zh, zr, d,
                 Tair='Tair', pressure='pressure', wind='wind', ustar='ustar', H='H',
                 z0m=None, stab_formulation='Dyer_1970',
                 Sc=None, Sc_name=None, constants=None):

    alpha = 4.39 - 3.97 * np.exp(-0.258 * LAI)
    estimate_z0m = z0m is None

    wind_zh = wind_profile(data, z=zh, Tair=Tair, pressure=pressure, ustar=ustar, H=H,
                           zr=zr, estimate_z0m=estimate_z0m, zh=zh, d=d, z0m=z0m,
                           frac_z0m=None, stab_correction=True, stab_formulation=stab_formulation)

    wind_zh = np.maximum(0.01, wind_zh)

    if Sc is not None or Sc_name is not None:
        if len(Sc) != len(Sc_name):
            raise ValueError("Arguments 'Sc' and 'Sc_name' must have the same length")
        if not isinstance(Sc, (list, np.ndarray)):
            raise TypeError("Argument 'Sc' must be numeric")

    Gb_h = LAI * ((0.02 / alpha) * np.sqrt(wind_zh / leafwidth) * (1 - np.exp(-alpha / 2)))
    Rb_h = 1 / Gb_h
    kB_h = Rb_h * constants['k'] * data[ustar]

    Sc_full = [constants['Sc_CO2']] + (Sc if Sc else [])
    Gb_x = {f"Gb_{name}": Gb_h / ((s / constants['Pr']) ** 0.67)
            for s, name in zip(Sc_full, ['CO2'] + (Sc_name if Sc_name else []))}

    return pd.DataFrame({'Gb_h': Gb_h, 'Rb_h': Rb_h, 'kB_h': kB_h, **Gb_x})


def gb_su(data, zh, zr, d, Dl,
          Tair='Tair', pressure='pressure', ustar='ustar', wind='wind', H='H',
          z0m=None, fc=None, LAI=None, N=2, Cd=0.2, hs=0.01,
          stab_formulation='Dyer_1970', Sc=None, Sc_name=None, constants=None):

    if fc is None:
        if LAI is None:
            raise ValueError("One of 'fc' or 'LAI' must be provided")
        fc = 1 - np.exp(-LAI / 2)

    estimate_z0m = z0m is None

    wind_zh = wind_profile(data, z=zh, Tair=Tair, pressure=pressure, ustar=ustar, H=H,
                           zr=zr, estimate_z0m=estimate_z0m, zh=zh, d=d, z0m=z0m,
                           frac_z0m=None, stab_correction=True, stab_formulation=stab_formulation)

    v = kinematic_viscosity(data[Tair], data[pressure], constants)
    Re = reynolds_number(data[Tair], data[pressure], data[ustar], hs, constants)
    kBs = 2.46 * (Re ** 0.25) - np.log(7.4)
    Reh = Dl * wind_zh / v
    Ct = constants['Pr'] ** -0.6667 * Reh ** -0.5 * N

    kB_h = (constants['k'] * Cd) / (4 * Ct * data[ustar] / wind_zh) * fc ** 2 + kBs * (1 - fc) ** 2
    Rb_h = kB_h / (constants['k'] * data[ustar])
    Gb_h = 1 / Rb_h

    if Sc is not None or Sc_name is not None:
        if len(Sc) != len(Sc_name):
            raise ValueError("Arguments 'Sc' and 'Sc_name' must have the same length")
        if not isinstance(Sc, (list, np.ndarray)):
            raise TypeError("Argument 'Sc' must be numeric")

    Sc_full = [constants['Sc_CO2']] + (Sc if Sc else [])
    Gb_x = {f"Gb_{name}": Gb_h / ((s / constants['Pr']) ** 0.67)
            for s, name in zip(Sc_full, ['CO2'] + (Sc_name if Sc_name else []))}

    return pd.DataFrame({'Gb_h': Gb_h, 'Rb_h': Rb_h, 'kB_h': kB_h, **Gb_x})


def roughness_length_heat(z0m, kB_h):
    return z0m / np.exp(kB_h)

# =========================================================================================================
# check_input.r

import pandas as pd
import numpy as np

def check_length(varlist):
    """
    Test Variables for Equal Length

    Parameters
    ----------
    varlist : list
        List of variables for which the length has to be compared.

    Returns
    -------
    list
        Validated list of variables
    """
    flat_list = []
    for v in varlist:
        if isinstance(v, list):
            flat_list.extend(v)
        else:
            flat_list.append(v)

    lengths = [len(v) for v in flat_list if hasattr(v, '__len__') and not isinstance(v, str)]

    unique_lengths = list(set(lengths))

    if len(unique_lengths) >= 2:
        if sorted(unique_lengths)[0] != 1 or len(unique_lengths) > 2:
            raise ValueError("All input variables must have the same length or a length of 1!")

    return flat_list

def check_input(data=None, **kwargs):
    """
    Check Input for Functions in the bigleaf Package (Python version)

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray or None
        Optional input dataset
    **kwargs : dict
        Input variables as keyword arguments

    Returns
    -------
    dict
        Dictionary of validated variables
    """
    vars_list = check_length(list(kwargs.values()))
    validated_vars = {}

    for varname, var in kwargs.items():
        if isinstance(var, str):
            if data is not None:
                if len(var) == 1:
                    if var in data.columns:
                        column = data[var]
                        if pd.api.types.is_numeric_dtype(column):
                            validated_vars[varname] = column.values
                        else:
                            raise TypeError(f"Column '{var}' representing '{varname}' in the input must be numeric.")
                    else:
                        raise ValueError(f"There is no column named '{var}' in the input data. "
                                         f"Provide a valid column name or a numeric array of correct length.")
                else:
                    raise ValueError(f"Variable name '{varname}' must have length 1.")
            else:
                raise ValueError(f"Variable '{var}' is a string and interpreted as a column name, "
                                 "but no input DataFrame was provided.")
        else:
            # Handle numeric or other directly provided input
            if var is None or (isinstance(var, float) and np.isnan(var)):
                validated_vars[varname] = var
                continue

            if data is not None:
                if isinstance(var, (list, np.ndarray, pd.Series)):
                    if len(var) == len(data):
                        validated_vars[varname] = np.array(var)
                    elif len(var) == 1:
                        validated_vars[varname] = np.repeat(var, len(data))
                    else:
                        raise ValueError(f"Variable '{varname}' must have same length as data or be of length 1.")
                else:
                    raise TypeError(f"Variable '{varname}' must be numeric.")
            else:
                if isinstance(var, (int, float, np.ndarray, list, pd.Series)):
                    validated_vars[varname] = np.array(var)
                else:
                    raise TypeError(f"Variable '{varname}' must be numeric.")

    return validated_vars

# =========================================================================================================

# datasets_description.r

"""
AT_Neu_Jul_2010: Half-hourly eddy covariance data from Neustift (Austria)

Description
-----------
Half-hourly eddy covariance measurements from the FLUXNET site AT-Neu, a mountain meadow in Austria.
Data are from July 2010.

Source: https://sites.fluxdata.org/AT-Neu/

Format
------
DataFrame with 1488 rows and 31 columns:

- year : int
    Year of measurement
- month : int
    Month of measurement
- doy : int
    Day of year
- hour : float
    Hour (0 to 23.5)
- Tair : float
    Air temperature (°C) [TA_F]
- Tair_qc : int
    Quality control flag for Tair [TA_F_QC]
- PPFD : float
    Photosynthetic photon flux density (μmol m⁻² s⁻¹) [PPFD_IN]
- PPFD_qc : int
    QC flag for PPFD [PPFD_IN_QC]
- VPD : float
    Vapor pressure deficit (kPa) [VPD_F]
- VPD_qc : int
    QC flag for VPD [VPD_F_QC]
- pressure : float
    Atmospheric pressure (kPa) [PA_F]
- precip : float
    Precipitation (mm) [P_F]
- precip_qc : int
    QC flag for precip [P_F_QC]
- ustar : float
    Friction velocity (m s⁻¹) [USTAR]
- wind : float
    Horizontal wind speed (m s⁻¹) [WS_F]
- wind_qc : int
    QC flag for wind [WS_F_QC]
- Ca : float
    CO₂ concentration (ppm) [CO2_F_MDS]
- Ca_qc : int
    QC flag for Ca [CO2_F_MDS_QC]
- LW_up : float
    Upward longwave radiation (W m⁻²) [LW_OUT]
- Rn : float
    Net radiation (W m⁻²) [NETRAD]
- LE : float
    Latent heat flux (W m⁻²) [LE_F_MDS]
- LE_qc : int
    QC flag for LE [LE_F_MDS_QC]
- H : float
    Sensible heat flux (W m⁻²) [H_F_MDS]
- H_qc : int
    QC flag for H [H_F_MDS_QC]
- G : float
    Ground heat flux (W m⁻²) [G_F_MDS]
- G_qc : int
    QC flag for G [G_F_MDS_QC]
- NEE : float
    Net ecosystem exchange (μmol m⁻² s⁻¹) [NEE_VUT_USTAR50]
- NEE_qc : int
    QC flag for NEE [NEE_VUT_USTAR50_QC]
- GPP : float
    Gross primary productivity (μmol m⁻² s⁻¹) [GPP_NT_VUT_USTAR50]
- GPP_qc : int
    QC flag for GPP [NEE_VUT_USTAR50_QC]
- Reco : float
    Ecosystem respiration (μmol m⁻² s⁻¹) [RECO_NT_VUT_USTAR50]

Notes
-----
Some units have been converted (e.g., VPD from hPa to kPa).
Original variable names from the FLUXNET2015 dataset are provided in square brackets.

Source
------
https://fluxnet.org/ (accessed 09 November 2016)
"""

"""
DE_Tha_Jun_2014: Half-hourly eddy covariance data from Tharandt (Germany)

Description
-----------
Measurements from DE-Tha, a spruce forest site in Eastern Germany. Data collected in June 2014.

Source: https://sites.fluxdata.org/DE-Tha/

Format
------
DataFrame with 1440 rows and 32 columns, similar to AT_Neu_Jul_2010, with the addition of:

- LW_down : float
    Downward longwave radiation (W m⁻²) [LW_IN_F]

Notes
-----
Units have been converted where necessary. Original FLUXNET2015 variable names are included in square brackets.

Source
------
https://fluxnet.org/ (accessed 09 November 2016)
"""

"""
FR_Pue_May_2012: Half-hourly eddy covariance data from Puechabon (France)

Description
-----------
Data from FR-Pue, a Mediterranean evergreen oak forest in Southern France. Measurements are from May 2012.

Source: https://sites.fluxdata.org/FR-Pue/

Format
------
DataFrame with 1488 rows and 29 columns.

Structure is mostly identical to AT_Neu_Jul_2010, but this dataset lacks `G`, `G_qc`, and `LW_down`.

Notes
-----
Units have been converted from original FLUXNET2015 values where necessary.

Source
------
https://fluxnet.org/ (accessed 09 November 2016)
"""
# =========================================================================================================

# decoupling.r

import numpy as np
import pandas as pd


def longwave_conductance(Tair, LAI, constants):
    """
    Longwave Radiative Transfer Conductance of the Canopy (Martin, 1989)

    Parameters
    ----------
    Tair : array-like or float
        Air temperature (°C)
    LAI : array-like or float
        Leaf area index (m2 m-2)
    constants : dict
        Must include:
            - 'Kelvin': conversion from °C to K
            - 'sigma': Stefan-Boltzmann constant (W m-2 K-4)
            - 'cp': specific heat of air for constant pressure (J K-1 kg-1)

    Returns
    -------
    Gr : ndarray or float
        Longwave radiative transfer conductance of the canopy (m s-1)
    """
    Tair_K = np.asarray(Tair) + constants['Kelvin']
    Gr = 4 * constants['sigma'] * Tair_K**3 * LAI / constants['cp']
    return Gr


def decoupling(data, Tair_col="Tair", pressure_col="pressure", Ga_col="Ga_h", Gs_col="Gs_ms",
               approach="Jarvis&McNaughton_1986", LAI=None, Esat_formula="Sonntag_1990",
               constants=None):
    """
    Canopy-Atmosphere Decoupling Coefficient 'Omega'

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing columns for Tair, pressure, Ga, Gs
    Tair_col, pressure_col, Ga_col, Gs_col : str
        Column names for respective variables
    approach : str
        Either 'Jarvis&McNaughton_1986' or 'Martin_1989'
    LAI : float or array-like, optional
        Leaf area index (m2 m-2), required if using 'Martin_1989'
    Esat_formula : str
        Method to compute esat slope; passed to Esat_slope
    constants : dict
        Dictionary with keys: Kelvin, cp, eps, sigma, Pa2kPa

    Returns
    -------
    Omega : np.ndarray
        Decoupling coefficient
    """
    # Input checks
    if constants is None:
        raise ValueError("constants dictionary must be provided.")

    required_cols = [Tair_col, pressure_col, Ga_col, Gs_col]
    if not all(col in data.columns for col in required_cols):
        raise ValueError("Missing required columns in input data.")

    # Retrieve input data
    Tair = data[Tair_col].values
    pressure = data[pressure_col].values
    Ga = data[Ga_col].values
    Gs = data[Gs_col].values

    # Calculate slope of saturation vapor pressure (Pa/K)
    Delta = Esat_slope(Tair, formula=Esat_formula, constants=constants)  # returns a vector
    gamma = psychrometric_constant(Tair, pressure, constants)            # returns a vector
    epsilon = Delta / gamma

    if approach == "Jarvis&McNaughton_1986":
        Omega = (epsilon + 1) / (epsilon + 1 + Ga / Gs)

    elif approach == "Martin_1989":
        if LAI is None:
            raise ValueError("LAI must be provided for Martin_1989 approach.")

        Gr = longwave_conductance(Tair, LAI, constants)
        Omega = (epsilon + 1 + Gr / Ga) / (epsilon + (1 + Ga / Gs) * (1 + Gr / Ga))

    else:
        raise ValueError("Invalid approach selected.")

    return Omega

# =========================================================================================================
# energy_balance.r

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def biochemical_energy(NEE, alpha=0.422):
    """
    Radiant energy absorbed in photosynthesis or heat release by respiration from NEE.
    """
    return alpha * -np.array(NEE)

def energy_use_efficiency(GPP, Rn, alpha=0.422):
    """
    Fraction of net radiation fixed by primary productivity.
    """
    GPP = np.array(GPP)
    Rn = np.array(Rn)
    Sp = biochemical_energy(-GPP, alpha)
    mask = np.isfinite(Sp) & np.isfinite(Rn)
    return np.sum(Sp[mask]) / np.sum(Rn[mask])

def energy_closure(data, Rn='Rn', G=None, S=None, LE='LE', H='H',
                   instantaneous=False, missing_G_as_NA=False, missing_S_as_NA=False):
    """
    Calculates the energy balance closure from energy flux terms.
    """
    df = data.copy()
    df['Rn'] = df[Rn]
    df['LE'] = df[LE]
    df['H'] = df[H]
    df['G'] = df[G] if G else 0
    df['S'] = df[S] if S else 0

    if not missing_G_as_NA and G:
        df['G'] = df['G'].fillna(0)
    elif G is None:
        print("Ground heat flux G is not provided and set to 0.")

    if not missing_S_as_NA and S:
        df['S'] = df['S'].fillna(0)
    elif S is None:
        print("Energy storage fluxes S are not provided and set to 0.")

    if instantaneous:
        EBR = (df['LE'] + df['H']) / (df['Rn'] - df['G'] - df['S'])
        return EBR.to_numpy()

    mask = df[['Rn', 'LE', 'H']].notna().all(axis=1)
    if G: mask &= df[[G]].notna().all(axis=1)
    if S: mask &= df[[S]].notna().all(axis=1)
    df = df[mask]
    n = len(df)

    sum_LE_H = np.sum(df['LE'] + df['H'])
    sum_Rn_G_S = np.sum(df['Rn'] - df['G'] - df['S'])
    EBR = sum_LE_H / sum_Rn_G_S

    X = (df['Rn'] - df['G'] - df['S']).values.reshape(-1, 1)
    y = (df['LE'] + df['H']).values
    model = LinearRegression().fit(X, y)

    return {
        "n": n,
        "intercept": round(model.intercept_, 3),
        "slope": round(model.coef_[0], 3),
        "r^2": round(model.score(X, y), 3),
        "EBR": round(EBR, 3)
    }

def isothermal_Rn(Rn, Tair, Tsurf, emissivity, constants=None):
    """
    Calculates the isothermal net radiation assuming surface and air at same temperature.
    """
    if constants is None:
        constants = {"sigma": 5.670374419e-8, "Kelvin": 273.15}
    Tair_K = np.array(Tair) + constants["Kelvin"]
    Tsurf_K = np.array(Tsurf) + constants["Kelvin"]
    return np.array(Rn) + emissivity * constants["sigma"] * (Tsurf_K**4 - Tair_K**4)

# =========================================================================================================

# evapotranspiration.r

import numpy as np
import pandas as pd

def potential_ET(data=None, Tair="Tair", pressure="pressure", Rn="Rn", G=None, S=None,
                 VPD="VPD", Ga="Ga", approach="Priestley-Taylor", alpha=1.26,
                 Gs_pot=0.6, missing_G_as_NA=False, missing_S_as_NA=False,
                 Esat_formula="Sonntag_1990", constants=None):

    if constants is None:
        constants = bigleaf_constants()

    if approach not in ["Priestley-Taylor", "Penman-Monteith"]:
        raise ValueError("approach must be either 'Priestley-Taylor' or 'Penman-Monteith'")

    # Extract data if necessary
    def extract(var):
        return data[var] if isinstance(var, str) else var

    Tair = extract(Tair)
    pressure = extract(pressure)
    Rn = extract(Rn)
    if G is not None:
        G = extract(G)
        if not missing_G_as_NA:
            G = np.nan_to_num(G)
    else:
        print("Ground heat flux G is not provided and set to 0.")
        G = 0

    if S is not None:
        S = extract(S)
        if not missing_S_as_NA:
            S = np.nan_to_num(S)
    else:
        print("Energy storage fluxes S are not provided and set to 0.")
        S = 0

    gamma = psychrometric_constant(Tair, pressure, constants)
    Delta = Esat_slope(Tair, Esat_formula, constants)["Delta"]

    if approach == "Priestley-Taylor":
        LE_pot = (alpha * Delta * (Rn - G - S)) / (Delta + gamma)
        ET_pot = LE_to_ET(LE_pot, Tair)

    elif approach == "Penman-Monteith":
        VPD = extract(VPD)
        Ga = extract(Ga)
        Gs_pot = mol_to_ms(Gs_pot, Tair=Tair, pressure=pressure, constants=constants)
        rho = air_density(Tair, pressure, constants)
        numerator = Delta * (Rn - G - S) + rho * constants['cp'] * VPD * Ga
        denominator = Delta + gamma * (1 + Ga / Gs_pot)
        LE_pot = numerator / denominator
        ET_pot = LE_to_ET(LE_pot, Tair)

    return pd.DataFrame({'ET_pot': ET_pot, 'LE_pot': LE_pot})

# =========================================================================================================
# filter_data.r

def filter_data(df,
                vars=None,
                filter_vars=True,
                quality_control=None,
                quality_threshold=0,
                precip=None,
                exclude_precip=True,
                u_star=None,
                u_star_threshold=None,
                growing_season=None,
                include_growing_season_only=True):
    """
    Filter eddy covariance data based on quality control, precipitation, u* threshold, and growing season.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with all variables as columns.
    vars : list of str, optional
        Variable names to include in the filtered output. If None, all columns are returned.
    filter_vars : bool
        If True, return only the filtered variables in `vars`; else, return all variables.
    quality_control : str or list of str, optional
        Column(s) with quality control flags (0, 1, 2). Can be one or multiple columns.
    quality_threshold : int
        Maximum quality control value to allow (0 = best).
    precip : str, optional
        Name of the precipitation column.
    exclude_precip : bool
        If True, exclude time steps with precipitation.
    u_star : str, optional
        Column name of friction velocity (u*).
    u_star_threshold : float or None
        Minimum acceptable u* value.
    growing_season : str, optional
        Name of column (bool or 0/1) indicating growing season.
    include_growing_season_only : bool
        If True, retain only growing season records.

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """

    data = df.copy()
    mask = pd.Series(True, index=data.index)

    # Quality control
    if quality_control is not None:
        if isinstance(quality_control, str):
            quality_control = [quality_control]
        for qc_col in quality_control:
            mask &= data[qc_col] <= quality_threshold

    # Precipitation
    if exclude_precip and precip is not None:
        mask &= data[precip] == 0

    # u* threshold
    if u_star is not None and u_star_threshold is not None:
        mask &= data[u_star] >= u_star_threshold

    # Growing season
    if growing_season is not None and include_growing_season_only:
        mask &= data[growing_season].astype(bool)

    filtered_data = data[mask]

    if vars is not None and filter_vars:
        filtered_data = filtered_data[vars]

    return filtered_data

# =========================================================================================================
# meteorological_variables.r

def air_density(Tair, pressure, constants = None):
    """Calculate air density."""
    if constants is None:
        constants = bigleaf_constants()
    Tair_K = Tair + constants['Kelvin']
    pressure_Pa = pressure * constants['kPa2Pa']
    return pressure_Pa / (constants['Rd'] * Tair_K)

def pressure_from_elevation(elev, Tair, VPD=None, constants=None, virtual_temp_func=None):
    """Calculate pressure from elevation."""
    if constants is None:
        constants = bigleaf_constants()
    Tair_K = Tair + constants['Kelvin']

    if VPD is None:
        pressure = constants['pressure0'] / np.exp(constants['g'] * elev / (constants['Rd'] * Tair_K))
    else:
        pressure1 = constants['pressure0'] / np.exp(constants['g'] * elev / (constants['Rd'] * Tair_K))
        Tv = virtual_temp_func(Tair, pressure1 * constants['Pa2kPa'], VPD, "Sonntag_1990", constants)
        Tv_K = Tv + constants['Kelvin']
        pressure = constants['pressure0'] / np.exp(constants['g'] * elev / (constants['Rd'] * Tv_K))

    return pressure * constants['Pa2kPa']

def esat_slope(Tair, formula="Sonntag_1990", constants=None):
    """Calculate saturation vapor pressure and its slope."""
    if constants is None:
        constants = bigleaf_constants()
    if formula == "Sonntag_1990":
        a, b, c = 611.2, 17.62, 243.12
    elif formula == "Alduchov_1996":
        a, b, c = 610.94, 17.625, 243.04
    elif formula == "Allen_1998":
        a, b, c = 610.8, 17.27, 237.3
    else:
        raise ValueError("Invalid formula name.")

    exp_term = np.exp((b * Tair) / (c + Tair))
    Esat = a * exp_term * constants['Pa2kPa']
    Delta = a * exp_term * (b / (c + Tair) - (b * Tair) / (c + Tair)**2) * constants['Pa2kPa']

    return pd.DataFrame({'Esat': Esat, 'Delta': Delta})

def latent_heat_vaporization(Tair):
    """Calculate latent heat of vaporization."""
    k1 = 2.501
    k2 = 0.00237
    return (k1 - k2 * Tair) * 1e6  # J/kg

def psychrometric_constant(Tair, pressure, constants = None):
    """Calculate psychrometric constant."""
    if constants is None:
        constants = bigleaf_constants()
    lambda_val = latent_heat_vaporization(Tair)
    return (constants['cp'] * pressure) / (constants['eps'] * lambda_val)

def wetbulb_solver(ea, Tair, gamma, accuracy, formula, constants = None):
    """Solve for wet bulb temperature."""
    if constants is None:
        constants = bigleaf_constants()
    def objective(Tw):
        Esat_val = esat_slope(Tw, formula, constants)['Esat']
        return np.abs(ea - (Esat_val - constants['Le067'] * gamma * (Tair - Tw)))

    result = minimize_scalar(objective, bounds=(-100, 100), method='bounded', options={'xatol': accuracy})
    return result

def wetbulb_temp(Tair, pressure, VPD, accuracy=1e-3, formula="Sonntag_1990", constants=None, vpd_to_e=None):
    """Calculate wet-bulb temperature."""
    if constants is None:
        constants = bigleaf_constants()
    if not isinstance(accuracy, (float, int)):
        raise ValueError("'accuracy' must be numeric")
    if accuracy > 1:
        print("'accuracy' is set to 1 degC")
        accuracy = 1

    ndigits = int(np.ceil(-np.log10(accuracy))) if accuracy < 1 else 0

    gamma = psychrometric_constant(Tair, pressure, constants)
    ea = vpd_to_e(VPD, Tair, formula, constants)

    Tw = []
    for i in range(len(Tair)):
        if any(np.isnan([ea[i], Tair[i], gamma[i]])) or any(np.isinf([ea[i], Tair[i], gamma[i]])):
            Tw.append(np.nan)
        else:
            res = wetbulb_solver(ea[i], Tair[i], gamma[i], accuracy, formula, constants)
            Tw.append(round(res.x, ndigits))

    return np.array(Tw)


def dew_point_solver(ea, accuracy, Esat_formula, constants = None):
    """
    Solves for the dew point temperature (Td) given actual vapor pressure (ea) by 
    minimizing the difference between ea and the saturation vapor pressure (Esat).

    Parameters:
    ----------
    ea : float
        Actual vapor pressure (Pa or hPa, depending on the Esat_slope output).
    accuracy : float
        Desired accuracy (tolerance) for the optimization (in °C).
    Esat_formula : str or callable
        The formula or method used to compute saturation vapor pressure.
    constants : dict
        A dictionary of physical constants used in the Esat_slope calculation.

    Returns:
    -------
    result : OptimizeResult
        The result object from scipy.optimize.minimize_scalar, containing the 
        optimized dew point temperature and additional metadata.
    """
    def objective(Td):
        Esat = esat_slope(Td, Esat_formula, constants)["Esat"]
        return abs(ea - Esat)
    
    result = minimize_scalar(objective, bounds=(-100, 100), method='bounded', options={'xatol': accuracy})
    return result


def dew_point(Tair, VPD, accuracy=1e-3, Esat_formula="Sonntag_1990", constants=None):
    """
    Calculates the dew point temperature from air temperature and vapor pressure deficit (VPD)
    using numerical optimization to solve for the temperature at which saturation vapor pressure 
    equals the actual vapor pressure.

    Parameters:
    ----------
    Tair : array-like
        Air temperature in degrees Celsius.
    VPD : array-like
        Vapor Pressure Deficit (same units as returned by `VPD_to_e`, typically Pa or hPa).
    accuracy : float, optional (default=1e-3)
        Desired accuracy (tolerance in °C) for dew point calculation.
        If greater than 1, it will be reset to 1 and a message will be printed.
    Esat_formula : str, optional (default="Sonntag_1990")
        The formula or method used to compute saturation vapor pressure (e.g., "Sonntag_1990").
    constants : dict, optional
        Dictionary of physical constants used in saturation vapor pressure calculations.
        If None, defaults will be used via `bigleaf_constants()`.

    Returns:
    -------
    Td : numpy.ndarray
        Dew point temperature array (°C) with same length as input `Tair`/`VPD`.
        If `ea` is missing or invalid, the corresponding Td value is NaN.
    """
    if constants is None:
        constants = bigleaf_constants()

    if not isinstance(accuracy, (int, float)):
        raise ValueError("'accuracy' must be numeric!")

    if accuracy > 1:
        print("'accuracy' is set to 1 degC")
        accuracy = 1

    # determine number of digits to print
    try:
        ndigits = int(str(accuracy).split('e-')[1])
    except IndexError:
        ndigits = 0

    ea = VPD_to_e(VPD, Tair, Esat_formula)

    Td = []
    for i in range(len(ea)):
        if ea[i] is None or isinstance(ea[i], float) and (np.isnan(ea[i]) or np.isinf(ea[i])):
            Td.append(np.nan)
        else:
            result = dew_point_solver(ea[i], accuracy=accuracy, Esat_formula=Esat_formula, constants=constants)
            Td.append(round(result['minimum'], ndigits))

    return np.array(Td)


def virtual_temp(Tair, pressure, VPD, formula, constants=None):
    """Calculate virtual temperature."""
    if constants is None:
        constants = bigleaf_constants()
    esat = esat_slope(Tair, formula, constants)['Esat']
    VPD_Pa = VPD * constants['kPa2Pa']
    return Tair * (1 + constants['eps'] * VPD_Pa / esat)

# def vpd_to_e(VPD, Tair, formula, constants=None):
#     """Convert VPD to actual vapor pressure (Pa)."""
#     if constants is None:
#         constants = bigleaf_constants()
#     esat = esat_slope(Tair, formula, constants)['Esat']
#     return VPD * esat * constants['kPa2Pa']

def kinematic_viscosity(Tair, pressure, constants):
    """
    Calculate kinematic viscosity of air.

    Parameters:
    - Tair: Air temperature in °C
    - pressure: Air pressure in kPa
    - constants: Dictionary of physical constants

    Returns:
    - Kinematic viscosity (m² s⁻¹)
    """
    Tair_K = Tair + constants["Kelvin"]
    pressure_Pa = pressure * constants["kPa2Pa"]

    v = 1.327e-05 * (constants["pressure0"] / pressure_Pa) * (Tair_K / constants["Tair0"]) ** 1.81
    return v


# =========================================================================================================
# optimum_temperature.r

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import Binarizer

def optimum_temperature(data, GPP="GPP", Tair="Tair", BLine=0.9, Obs_filter=30):
    # Check input
    if GPP not in data.columns or Tair not in data.columns:
        raise ValueError(f"Columns {GPP} and {Tair} must be present in the data")

    # Round to 1°C temperature bins
    Tair_bin = np.floor(data[Tair] + np.sign(data[Tair]) * 0.5).astype(int)

    # Get boundary line using quantiles
    df_bl = data.groupby(Tair_bin)[GPP].quantile(BLine).reset_index(name='GPP_Bline')

    # Get the number of observations in each temperature bin
    n_obs = data.groupby(Tair_bin).size().reset_index(name='n_obs')

    # Merge the data frames
    df_bl = pd.merge(df_bl, n_obs, on=Tair_bin)

    # Remove temperature bins with n_obs below the threshold
    df_bl = df_bl[df_bl['n_obs'] >= Obs_filter]

    # Get the smoothed boundary line using loess (locally weighted scatterplot smoothing)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(df_bl[[Tair_bin]])
    model = LinearRegression().fit(X_poly, df_bl['GPP_Bline'])

    # Predict smoothed values
    df_bl['GPP_Bline_smooth'] = model.predict(X_poly)

    # Find the thermal optimum (Topt)
    df_bl_sorted = df_bl.sort_values(by='GPP_Bline_smooth', ascending=False)
    Topt = df_bl_sorted.iloc[0]['Tair_bin']
    GPP_bl = df_bl_sorted.iloc[0]['GPP_Bline']

    opt_temp = {"Topt": Topt, "GPP_bl": GPP_bl}

    # Return results
    optimum_temp = {"df_bl": df_bl, "opt_temp": opt_temp}
    return optimum_temp

# =========================================================================================================
# potential_radiation.r

import numpy as np

def extraterrestrial_radiation(doy, solar_constant=1361):
    """
    Compute the extraterrestrial solar radiation with eccentricity correction.

    Parameters:
    doy (array-like): Day of year (DoY)
    solar_constant (float): Solar constant (W m^-2)

    Returns:
    np.ndarray: Extraterrestrial radiation (W m^-2)
    """
    # Fractional year in radians
    FracYearRad = 2 * np.pi * (doy - 1) / 365.24

    # Eccentricity correction
    ExtRadiation = solar_constant * (
        1.00011 + 0.034221 * np.cos(FracYearRad) + 0.00128 * np.sin(FracYearRad)
        + 0.000719 * np.cos(2 * FracYearRad) + 0.000077 * np.sin(2 * FracYearRad)
    )

    return ExtRadiation


def potential_radiation(doy, hour, latDeg, longDeg, timezone, useSolartime=True):
    """
    Compute potential radiation for a given geolocation and day of year.

    Parameters:
    doy (array-like): Day of year (start at 1)
    hour (array-like): Daytime as decimal hour of local time zone
    latDeg (float): Latitude (decimal degrees)
    longDeg (float): Longitude (decimal degrees)
    timezone (int): Time zone (hours)
    useSolartime (bool): If True, correct for solar time. Otherwise, use local winter time

    Returns:
    np.ndarray: Potential radiation (W m^-2)
    """
    # Assuming you have a function `compute_sun_position_doy_hour` to get solar elevation
    solElevRad = compute_sun_position_doy_hour(doy, hour, latDeg, longDeg, timezone, useSolartime)["elevation"]

    # Compute extraterrestrial radiation
    extRadiation = extraterrestrial_radiation(doy)

    # Calculate potential radiation
    potRad = np.where(solElevRad <= 0, 0, extRadiation * np.sin(solElevRad))

    return potRad

# Helper function: this function should be defined based on your solar time computation
def compute_sun_position_doy_hour(doy, hour, latDeg, longDeg, timezone, isCorrectSolartime=True):
    # Placeholder for actual implementation of sun position computation
    # This should return a DataFrame or structured array with 'elevation' data
    pass

# =========================================================================================================
# stability_correction.r

import numpy as np
import pandas as pd

# Constants (example values, adjust according to your constants)
def bigleaf_constants():
    return {
        'Kelvin': 273.15,  # Celsius to Kelvin conversion
        'cp': 1005,        # Specific heat of air at constant pressure (J/kg·K)
        'k': 0.4,          # von Karman constant
        'g': 9.81          # Gravitational acceleration (m/s^2)
    }

# Helper function to check if the necessary input columns are in the data
def check_input(data, required_columns):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Air density function
def air_density(Tair, pressure, constants):
    # Ideal gas law: rho = p / (R * T)
    R = 287.05  # Specific gas constant for dry air in J/(kg·K)
    return pressure * 1e3 / (R * Tair)  # converting pressure from kPa to Pa

# Monin-Obukhov Length function
def monin_obukhov_length(data, Tair="Tair", pressure="pressure", ustar="ustar", H="H", constants=None):
    if constants is None:
        constants = bigleaf_constants()

    check_input(data, [Tair, pressure, ustar, H])

    rho = air_density(data[Tair], data[pressure], constants)
    Tair_K = data[Tair] + constants['Kelvin']  # Celsius to Kelvin
    MOL = - (rho * constants['cp'] * data[ustar]**3 * Tair_K) / (constants['k'] * constants['g'] * data[H])

    return MOL

# Stability Parameter "zeta" function
def stability_parameter(data, Tair="Tair", pressure="pressure", ustar="ustar", H="H", zr="zr", d="d", constants=None):
    if constants is None:
        constants = bigleaf_constants()

    check_input(data, [Tair, pressure, ustar, H, zr, d])

    MOL = monin_obukhov_length(data, Tair, pressure, ustar, H, constants)
    zeta = (data[zr] - data[d]) / MOL

    return zeta

# Stability Correction function for Heat and Momentum
def stability_correction(zeta, formulation="Dyer_1970"):
    if formulation not in ["Dyer_1970", "Businger_1971"]:
        raise ValueError("Formulation must be either 'Dyer_1970' or 'Businger_1971'")

    psi_h = np.full_like(zeta, np.nan)
    psi_m = np.full_like(zeta, np.nan)

    if formulation == "Businger_1971":
        x_h = -7.8
        x_m = -6
        y_h = 0.95 * (1 - 11.6 * zeta) ** 0.5
        y_m = (1 - 19.3 * zeta) ** 0.25
    elif formulation == "Dyer_1970":
        x_h = x_m = -5
        y_h = (1 - 16 * zeta) ** 0.5
        y_m = (1 - 16 * zeta) ** 0.25

    # Stable conditions
    stable = zeta >= 0
    psi_h[stable] = x_h * zeta[stable]
    psi_m[stable] = x_m * zeta[stable]

    # Unstable conditions
    unstable = zeta < 0
    psi_h[unstable] = 2 * np.log((1 + y_h[unstable]) / 2)
    psi_m[unstable] = (2 * np.log((1 + y_m[unstable]) / 2) +
                       np.log((1 + y_m[unstable]**2) / 2) -
                       2 * np.arctan(y_m[unstable]) + np.pi / 2)

    return pd.DataFrame({'psi_h': psi_h, 'psi_m': psi_m})

# # Example usage:
# data = pd.DataFrame({
#     'Tair': [25],
#     'pressure': [100],
#     'ustar': [0.3],
#     'H': [100],
#     'zr': [40],
#     'd': [15]
# })

# zeta_values = stability_parameter(data, zr="zr", d="d")
# corrections = stability_correction(zeta_values)
# print(corrections)

# =========================================================================================================
# surface_conditions.r

import numpy as np
import pandas as pd

def surface_conditions(Tair, pressure, H, LE, VPD, Ga,
                       calc_surface_CO2=False, Ca=None, NEE=None, Ga_CO2=None,
                       Esat_formula="Sonntag_1990", constants=None):
    """
    Calculate big-leaf surface conditions (temperature, vapor pressure, humidity, CO2).
    """
    if constants is None:
        raise ValueError("Constants must be provided.")

    rho = air_density(Tair, pressure, constants)
    gamma = psychrometric_constant(Tair, pressure, constants)

    Tsurf = Tair + H / (rho * constants["cp"] * Ga)

    esat = esat_slope(Tair, Esat_formula, constants)["Esat"]
    e = esat - VPD
    esat_surf = esat_slope(Tsurf, Esat_formula, constants)["Esat"]
    esurf = e + (LE * gamma) / (Ga * rho * constants["cp"])
    VPD_surf = np.maximum(esat_surf - esurf, 0)
    qsurf = VPD_to_q(VPD_surf, Tsurf, pressure, Esat_formula, constants)
    rH_surf = VPD_to_rH(VPD_surf, Tsurf, Esat_formula)

    if calc_surface_CO2:
        if Ca is None or NEE is None or Ga_CO2 is None:
            raise ValueError("Ca, NEE, and Ga_CO2 must be provided if calc_surface_CO2 is True.")
        Ca_surf = surface_CO2(Ca, NEE, Ga_CO2, Tair, pressure)
    else:
        Ca_surf = np.full_like(np.array(Tair), np.nan)

    return pd.DataFrame({
        "Tsurf": Tsurf,
        "esat_surf": esat_surf,
        "esurf": esurf,
        "VPD_surf": VPD_surf,
        "qsurf": qsurf,
        "rH_surf": rH_surf,
        "Ca_surf": Ca_surf
    })

def surface_CO2(Ca, NEE, Ga_CO2, Tair, pressure):
    """
    Calculate CO2 concentration at the canopy surface.
    """
    Ga_CO2_mol = ms_to_mol(Ga_CO2, Tair, pressure)
    return Ca + NEE / Ga_CO2_mol

def radiometric_surface_temp(LW_up, LW_down, emissivity, constants):
    """
    Calculate radiometric surface temperature from longwave radiation.
    """
    Trad_K = ((LW_up - (1 - emissivity) * LW_down) / (constants["sigma"] * emissivity)) ** 0.25
    Trad_degC = Trad_K - constants["Kelvin"]
    return pd.DataFrame({"Trad_K": Trad_K, "Trad_degC": Trad_degC})

# =========================================================================================================
# surface_conductance.r

import numpy as np
import pandas as pd

def surface_conductance(data, Tair="Tair", pressure="pressure", Rn="Rn", G=None, S=None,
                        VPD="VPD", LE="LE", Ga="Ga_h", missing_G_as_NA=False, missing_S_as_NA=False,
                        formulation="Penman-Monteith", Esat_formula="Sonntag_1990",
                        constants=bigleaf_constants()):

    formulation = formulation if formulation in ["Penman-Monteith", "Flux-Gradient"] else "Penman-Monteith"

    if formulation == "Flux-Gradient":
        check_input(data, [Tair, pressure, VPD, LE])

        Gs_mol = (LE_to_ET(data[LE], data[Tair], constants) / constants["Mw"]) * data[pressure] / data[VPD]
        Gs_ms = mol_to_ms(Gs_mol, data[Tair], data[pressure], constants)

    elif formulation == "Penman-Monteith":
        check_input(data, [Tair, pressure, VPD, LE, Rn, Ga])

        G_col = data[G] if G is not None else 0
        if G is not None and not missing_G_as_NA:
            G_col = G_col.fillna(0)

        S_col = data[S] if S is not None else 0
        if S is not None and not missing_S_as_NA:
            S_col = S_col.fillna(0)

        Delta = Esat_slope(data[Tair], Esat_formula, constants)["Delta"]
        gamma = psychrometric_constant(data[Tair], data[pressure], constants)
        rho = air_density(data[Tair], data[pressure], constants)

        Rn_net = data[Rn] - G_col - S_col
        Ga_col = data[Ga]
        VPD_col = data[VPD]
        LE_col = data[LE]

        numerator = LE_col * Ga_col * gamma
        denominator = Delta * Rn_net + rho * constants["cp"] * Ga_col * VPD_col - LE_col * (Delta + gamma)

        Gs_ms = numerator / denominator
        Gs_mol = ms_to_mol(Gs_ms, data[Tair], data[pressure], constants)

    return pd.DataFrame({"Gs_ms": Gs_ms, "Gs_mol": Gs_mol})

# =========================================================================================================

# surface_roughness.r

import numpy as np
import pandas as pd

def reynolds_number(Tair, pressure, ustar, z0m, constants):
    v = kinematic_viscosity(Tair, pressure, constants)
    Re = z0m * ustar / v
    return Re

def roughness_parameters(method, zh, frac_d=0.7, frac_z0m=0.1, LAI=None, zr=None, cd=0.2, hs=0.01,
                         data=None, Tair_col="Tair", pressure_col="pressure", wind_col="wind",
                         ustar_col="ustar", H_col="H", d=None, z0m=None,
                         stab_roughness=True, stab_formulation="Dyer_1970", constants=None):

    if constants is None:
        constants = bigleaf_constants()

    if method == "canopy_height":
        d = frac_d * zh
        z0m = frac_z0m * zh
        z0m_se = np.nan

    elif method == "canopy_height&LAI":
        X = cd * LAI
        d = 1.1 * zh * np.log(1 + X**0.25)
        if 0 <= X <= 0.2:
            z0m = hs + 0.3 * X**0.5
        else:
            z0m = 0.3 * zh * (1 - d / zh)
        z0m_se = np.nan

    elif method == "wind_profile":
        check_input(data, [Tair_col, pressure_col, wind_col, ustar_col, H_col])

        if d is None:
            d = frac_d * zh

        wind = data[wind_col]
        ustar = data[ustar_col]

        if stab_roughness:
            zeta = stability_parameter(data, Tair_col, pressure_col, ustar_col, H_col, zr, d, constants)
            psi_m = stability_correction(zeta, formulation=stab_formulation)["psi_m"]
            z0m_all = (zr - d) * np.exp(-constants['k'] * wind / ustar - psi_m)
        else:
            z0m_all = (zr - d) * np.exp(-constants['k'] * wind / ustar)

        z0m_all[z0m_all > zh] = np.nan
        z0m = np.nanmedian(z0m_all)
        valid = z0m_all.dropna()
        z0m_se = constants['se_median'] * (np.std(valid) / np.sqrt(len(valid)))

    return pd.DataFrame({"d": [d], "z0m": [z0m], "z0m_se": [z0m_se]})

def wind_profile(data, z, Tair_col="Tair", pressure_col="pressure", ustar_col="ustar", H_col="H",
                 wind_col="wind", zr=None, zh=None, d=None, frac_d=0.7, z0m=None, frac_z0m=0.1,
                 estimate_z0m=True, stab_correction=True, stab_formulation="Dyer_1970",
                 constants=None):

    if constants is None:
        constants = bigleaf_constants()

    check_input(data, [ustar_col])

    if d is None:
        d = frac_d * zh

    if estimate_z0m:
        rough = roughness_parameters("wind_profile", zh=zh, zr=zr, frac_d=frac_d, data=data,
                                     Tair_col=Tair_col, pressure_col=pressure_col, wind_col=wind_col,
                                     ustar_col=ustar_col, H_col=H_col, stab_roughness=stab_correction,
                                     stab_formulation=stab_formulation, constants=constants)
        z0m = rough["z0m"].values[0]

    z_diff = z - d
    z_diff[z_diff <= z0m] = np.nan

    if stab_correction:
        zeta = stability_parameter(data, Tair_col, pressure_col, ustar_col, H_col, zr, d, constants)
        psi_m = stability_correction(zeta, formulation=stab_formulation)["psi_m"]
    else:
        psi_m = 0

    ustar = data[ustar_col]
    uz = (ustar / constants['k']) * (np.log(z_diff / z0m) - psi_m)
    uz[z_diff <= z0m] = 0

    return uz
# =========================================================================================================
# unit_conversions.r

# Unit conversions - Python version of R's bigleaf conversion functions

def LE_to_ET(LE, Tair):
    lambda_ = latent_heat_vaporization(Tair)
    return LE / lambda_

def ET_to_LE(ET, Tair):
    lambda_ = latent_heat_vaporization(Tair)
    return ET * lambda_

def ms_to_mol(G_ms, Tair, pressure, constants):
    Tair_K = Tair + constants['Kelvin']
    pressure_Pa = pressure * constants['kPa2Pa']
    return G_ms * pressure_Pa / (constants['Rgas'] * Tair_K)

def mol_to_ms(G_mol, Tair, pressure, constants):
    Tair_K = Tair + constants['Kelvin']
    pressure_Pa = pressure * constants['kPa2Pa']
    return G_mol * (constants['Rgas'] * Tair_K) / pressure_Pa

def VPD_to_rH(VPD, Tair, Esat_slope, Esat_formula, constants):
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    return 1 - VPD / esat

def rH_to_VPD(rH, Tair, Esat_slope, Esat_formula, constants):
    if any(r > 1 for r in rH if r is not None):
        print("Warning: relative humidity (rH) has to be between 0 and 1.")
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    return esat - rH * esat

def e_to_rH(e, Tair, Esat_slope, Esat_formula, constants):
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    if any(val > es + 1e-15 for val, es in zip(e, esat) if val is not None):
        print("Warning: Provided vapour pressure was higher than saturation. Returning rH=1 for those cases.")
    return [min(1, val/es) for val, es in zip(e, esat)]

def VPD_to_e(VPD, Tair, Esat_slope, Esat_formula, constants):
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    return esat - VPD

def e_to_VPD(e, Tair, Esat_slope, Esat_formula, constants):
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    return esat - e

def e_to_q(e, pressure, constants):
    return constants['eps'] * e / (pressure - (1 - constants['eps']) * e)

def q_to_e(q, pressure, constants):
    return q * pressure / ((1 - constants['eps']) * q + constants['eps'])

def q_to_VPD(q, Tair, pressure, Esat_slope, Esat_formula, constants):
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    e = q_to_e(q, pressure, constants)
    return esat - e

def VPD_to_q(VPD, Tair, pressure, Esat_slope, Esat_formula, constants):
    esat = Esat_slope(Tair, Esat_formula, constants)["Esat"]
    e = esat - VPD
    return e_to_q(e, pressure, constants)

def Rg_to_PPFD(Rg, J_to_mol=4.6, frac_PAR=0.5):
    return Rg * frac_PAR * J_to_mol

def PPFD_to_Rg(PPFD, J_to_mol=4.6, frac_PAR=0.5):
    return PPFD / (frac_PAR * J_to_mol)

def kg_to_mol(mass, molar_mass):
    return mass / molar_mass

def umolCO2_to_gC(CO2_flux, constants):
    return CO2_flux * constants['umol2mol'] * constants['Cmol'] * constants['kg2g'] * constants['days2seconds']

def gC_to_umolCO2(C_flux, constants):
    return (C_flux * constants['g2kg'] / constants['days2seconds']) / constants['Cmol'] * constants['mol2umol']

# =========================================================================================================