import numpy as np

def cal_vpd(Tair, RH):
    # See: Murray FW (1967) On the computation of saturation vapor pressure. J. Appl. Meteorol. 6: 203-204.
    # Monteith JL, Unsworth MH (1990) Principles of environmental physics. Arnold.
    # unit of Tair should be C.
    # saturated vapour pressure (SVP)
    SVP = 0.61078 * (np.e ** ((17.27 * Tair) / (Tair + 237.3)))
    VPD = SVP * (1 - RH / 100) * 100
    # VPD = 0.61078 * (np.e ** ((17.27 * Tair) / (Tair + 237.3))) * (1 - RH / 100) * 100
    return VPD