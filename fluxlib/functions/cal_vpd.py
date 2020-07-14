import numpy as np

def cal_vpd(Tair, RH):
    # unit of Tair should be C.
    VPD = 0.61078 * (np.e ** ((17.27 * Tair) / (Tair + 237.3))) * (1 - RH)
    return VPD