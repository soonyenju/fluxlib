"""
Converting bigleaf in r to Python
Credit: https://github.com/cran/bigleaf/tree/master/R
It's still under debugging, use with cautions
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

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