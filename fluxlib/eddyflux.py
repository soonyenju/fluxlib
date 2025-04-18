import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# --- NAMED TUPLES FOR STRUCTURED OUTPUT ---
PlanarFitCoefs = namedtuple("PlanarFitCoefs", ["k_vct", "tilt_coefs", "phi", "theta"])
RotatedVectors = namedtuple("RotatedVectors", ["rotated", "phi_theta"])
VickersMahrt = namedtuple("VickersMahrt", ["x", "nspikes", "ntrends", "kclass"])

# Supported 3D wind vector rotation methods
_VECTOR_ROTATION_METHODS = {"DR", "TR", "PF"}

def despike(series, n_std=3):
    """
    Remove spikes in a time series based on standard deviation thresholding.
    This version also returns the number of spikes detected.

    Parameters:
    -----------
    series : np.ndarray
        Input time series data.
    n_std : float
        Number of standard deviations for spike threshold.

    Returns:
    --------
    VickersMahrt : namedtuple
        - x: despiked time series
        - nspikes: number of spikes detected
        - ntrends: number of trends detected
        - kclass: classification (e.g., "Noisy" or "Trendy")
    """
    mean = np.mean(series)
    std = np.std(series)
    mask = np.abs(series - mean) > n_std * std
    despiked = series.copy()
    despiked[mask] = np.nan

    # Fill NaNs by interpolation (or use np.nanmean as fallback)
    nans = np.isnan(despiked)
    if np.any(nans):
        despiked[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), despiked[~nans])

    # Count spikes
    n_spikes = np.sum(mask)

    # Trend detection (simple approach: count significant changes)
    n_trends = np.sum(np.diff(despiked) > 0.1)  # placeholder for trend detection

    # Classification: classify as "Noisy" if there are many spikes, else "Trendy"
    kclass = "Noisy" if n_spikes > 10 else "Trendy"

    # Return as VickersMahrt tuple
    return VickersMahrt(despiked, n_spikes, n_trends, kclass)


# --- LAG CORRECTION ---
def cross_correlation_lag(x, y, max_lag=10):
    """
    Calculate the lag (shift) that maximizes the cross-correlation between two time series.

    This function computes the cross-correlation between two signals `x` and `y` for a range of lags, 
    and returns the lag that produces the highest correlation. This is useful for determining the time 
    shift that maximizes the relationship between two signals, often used in time series alignment or 
    lag-correction tasks.

    Parameters:
    -----------
    x : np.ndarray
        First time series data (1D array).
    y : np.ndarray
        Second time series data (1D array).
    max_lag : int, optional, default=10 (1s if 10Hz)
        Maximum lag (positive and negative) to consider for cross-correlation. The function will check
        lags in the range of `[-max_lag, ..., max_lag]`.

    Returns:
    --------
    int
        The optimal lag (in number of samples) that maximizes the cross-correlation between `x` and `y`.
        A positive value indicates that `x` leads `y`, and a negative value indicates that `y` leads `x`.

    Example:
    --------
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 4, 3, 2, 1])
    lag = cross_correlation_lag(x, y, max_lag=2)
    print(lag)  # Output: -2
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = [np.corrcoef(x[max_lag:-max_lag], np.roll(y, lag)[max_lag:-max_lag])[0, 1] for lag in lags]
    return lags[np.argmax(corrs)]

def apply_lag_correction(w, scalars, max_lag=10):
    """
    Apply lag correction to scalar time series using vertical wind.

    This function uses cross-correlation to determine the optimal lag between 
    the vertical wind (`w`) and each scalar time series in `scalars`. The scalar 
    time series are then shifted by the corresponding lag to align them with the 
    vertical wind.

    Parameters:
    -----------
    w : np.ndarray
        1D array of vertical wind time series.
    scalars : list of np.ndarray
        List of scalar time series (e.g., temperature, CO2, H2O, etc.) to be 
        lag-corrected.
    max_lag : int, optional
        The maximum number of lags (positive and negative) to check during 
        cross-correlation. Default is 10.

    Returns:
    --------
    list of np.ndarray
        A list of lag-corrected scalar time series. Each series is shifted 
        by the optimal lag to best align with the vertical wind (`w`).

    Notes:
    ------
    The lag correction is performed by computing the cross-correlation 
    between `w` and each scalar series, then applying the optimal lag 
    determined by the cross-correlation. The time series are shifted 
    accordingly using `np.roll`.
    """
    corrected = []
    for scalar in scalars:
        lag = cross_correlation_lag(w, scalar, max_lag)
        corrected.append(np.roll(scalar, -lag))
    return corrected

# --- VECTOR ROTATION METHODS ---
def planarfit(vectors):
    """
    Compute the tilt correction for a 3D vector field using the Planar Fit method.

    This function estimates the orientation of the best-fit plane to the given 3D vectors 
    (u, v, w) by solving a system of linear equations. The output is a unit vector normal to 
    the plane, which represents the vertical direction after tilt correction.

    Parameters:
    -----------
    vectors : np.ndarray
        A Nx3 array of 3D vectors, where each row represents a vector (u, v, w) in the 
        horizontal (u, v) and vertical (w) directions.

    Returns:
    --------
    np.ndarray : 
        A 3-element array representing the unit normal vector to the best-fit plane, 
        i.e., the corrected vertical direction after planar fit tilt correction. 
        The vector is oriented in the vertical (w) direction.
    
    Notes:
    ------
    The Planar Fit method assumes the input vectors (u, v, w) are from a coordinate system 
    that includes a vertical component (w), which will be aligned to the normal vector 
    after tilt correction. This method is commonly used in eddy covariance flux measurements 
    to correct for the tilt of the sensor.
    """
    u, v, w = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    n = vectors.shape[0]
    sum_u, sum_v, sum_w = np.sum(u), np.sum(v), np.sum(w)
    dot_uv, dot_uw, dot_vw = np.dot(u, v), np.dot(u, w), np.dot(v, w)
    dot_u2, dot_v2 = np.dot(u, u), np.dot(v, v)

    H = np.array([[n, sum_u, sum_v],
                  [sum_u, dot_u2, dot_uv],
                  [sum_v, dot_uv, dot_v2]])
    g = np.array([sum_w, dot_uw, dot_vw])
    tilt_coef = np.linalg.solve(H, g)

    k_2 = 1 / np.sqrt(1 + tilt_coef[1]**2 + tilt_coef[2]**2)
    k_0, k_1 = -tilt_coef[1] * k_2, -tilt_coef[2] * k_2
    return np.array([k_0, k_1, k_2])

def rotate_wind3d(wind3D, method="PF"):
    """
    Rotate wind vectors using specified method.

    Parameters:
    -----------
    wind3D : np.ndarray
        Nx3 array of (u, v, w)
    method : str
        Rotation method: 'PF' (Planar Fit), 'DR', or 'TR'

    Returns:
    --------
    RotatedVectors : namedtuple
        Rotated wind and placeholder for angles.
    """
    if method not in _VECTOR_ROTATION_METHODS:
        raise ValueError(f"Unsupported rotation method '{method}'. Supported: {_VECTOR_ROTATION_METHODS}")

    if method == "PF":
        k_vct = planarfit(wind3D)
        j_vct = np.cross(k_vct, np.mean(wind3D, axis=0))
        j_vct /= np.linalg.norm(j_vct)
        i_vct = np.cross(j_vct, k_vct)
        rotation_matrix = np.column_stack((i_vct, j_vct, k_vct))
        rotated = wind3D @ rotation_matrix.T
        return RotatedVectors(rotated=rotated, phi_theta=None)
    else:
        raise NotImplementedError(f"Rotation method '{method}' not implemented yet.")

# --- FLUX CALCULATION ---
def compute_ec_fluxes(w, T, CO2, H2O, P, u, v, rotation_method="PF"):
    """
    Compute eddy covariance fluxes with despiking, lag correction, and 3D rotation for 30-min period of data.

    Parameters:
    -----------
    w, T, CO2, H2O, P, u, v : np.ndarray
        Time series of vertical wind, temperature, CO2, H2O, pressure, and horizontal wind.
    rotation_method : str
        3D wind rotation method ('PF' supported).

    Returns:
    --------
    dict : Fluxes {'CO2_flux', 'H', 'LE'}
    """
    # Despike input signals and get VickersMahrt results
    w, T, CO2, H2O, P, u, v = [despike(series).x for series in [w, T, CO2, H2O, P, u, v]]

    # # Example of using the VickersMahrt results
    # for signal, name in zip([w, T, CO2, H2O, P, u, v], ["w", "T", "CO2", "H2O", "P", "u", "v"]):
    #     print(f"{name} - Spikes Detected: {signal.nspikes}, Trends: {signal.ntrends}, Class: {signal.kclass}")

    # Combine wind components and rotate
    wind3D = np.column_stack((u, v, w))
    rotated = rotate_wind3d(wind3D, method=rotation_method).rotated
    w_rot = rotated[:, 2]

    # Constants
    R = 8.314      # J/mol/K
    cp = 1005      # J/kg/K
    M_d = 0.02897  # kg/mol
    lambda_v = 2.45e6  # J/kg
    T_k = T + 273.15

    def prime(x): return x - np.mean(x)

    # Lag correction
    CO2, H2O = apply_lag_correction(w_rot, [CO2, H2O], max_lag=15)

    # Fluctuations
    w_prime = prime(w_rot)
    T_prime = prime(T_k)
    CO2_prime = prime(CO2)
    H2O_prime = prime(H2O)

    # Air density
    rho_air = P / (R * T_k)
    rho_air_mean = np.mean(rho_air)
    rho_air_kg = rho_air_mean * M_d

    # Sensible heat flux
    H = rho_air_kg * cp * np.mean(w_prime * T_prime)

    # Latent heat flux
    q = H2O / (1 + H2O)
    q_prime = prime(q)
    wq = np.mean(w_prime * q_prime)
    LE = rho_air_kg * lambda_v * wq

    # CO2 flux with WPL correction
    T_, CO2_, q_ = np.mean(T_k), np.mean(CO2), np.mean(q)
    wpl_term1 = (CO2_ / T_) * np.mean(w_prime * T_prime)
    wpl_term2 = (CO2_ / (1 - q_)) * wq
    CO2_flux = np.mean(w_prime * CO2_prime) + wpl_term1 + wpl_term2

    return {"CO2_flux": CO2_flux, "H": H, "LE": LE}

"""
# --- TEST WITH SIMULATED DATA ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 300
    u = np.random.normal(1.2, 0.1, n)
    v = np.random.normal(0.9, 0.1, n)
    w = np.random.normal(0.3, 0.05, n)
    T = np.random.normal(25, 0.3, n)
    P = np.random.normal(101325, 50, n)
    CO2 = np.roll(w + np.random.normal(0, 0.1, n), 5) + 400
    H2O = np.roll(w + np.random.normal(0, 0.1, n), -3) + 10

    result = compute_ec_fluxes(w=w, T=T, CO2=CO2, H2O=H2O, P=P, u=u, v=v, rotation_method="PF")

    print("\nEddy Covariance Fluxes (with despike, lag correction, and rotation):")
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
"""

def compute_spectral_density(signal, fs=1.0):
    """
    Compute the power spectral density of a signal using FFT.

    Parameters:
    -----------
    signal : np.ndarray
        Input time series data.
    fs : float
        Sampling frequency (samples per second).

    Returns:
    --------
    f : np.ndarray
        Frequencies.
    psd : np.ndarray
        Power spectral density (PSD).
    """
    # Perform FFT
    n = len(signal)
    fft_result = np.fft.fft(signal)
    psd = np.abs(fft_result)**2 / n  # Power spectral density
    f = np.fft.fftfreq(n, d=1/fs)  # Frequency bins

    # Only take the positive frequencies
    f = f[:n//2]
    psd = psd[:n//2]

    return f, psd

"""
# --- TEST WITH SIMULATED DATA ---

# Simulate some data (e.g., vertical wind and temperature)
np.random.seed(42)
n = 1024  # Number of samples
fs = 10  # Sampling frequency (Hz)
w = np.random.normal(0, 1, n)  # Simulated vertical wind

# Compute the spectral density
frequencies, psd = compute_spectral_density(w, fs)

# Plot the result
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.title('Power Spectral Density of Vertical Wind')
plt.grid(True)
plt.show()
"""

def compute_cospectral_density(signal1, signal2, fs=1.0):
    """
    Compute the cross spectral density (cospectrum) between two signals.

    Parameters:
    -----------
    signal1, signal2 : np.ndarray
        Input time series data.
    fs : float
        Sampling frequency (samples per second).

    Returns:
    --------
    f : np.ndarray
        Frequencies.
    cospec : np.ndarray
        Cospectral density.
    """
    # Perform FFT on both signals
    n = len(signal1)
    fft_signal1 = np.fft.fft(signal1)
    fft_signal2 = np.fft.fft(signal2)
    
    # Compute the cross-spectrum (complex conjugate of signal2's FFT)
    cospec = np.real(fft_signal1 * np.conj(fft_signal2)) / n
    f = np.fft.fftfreq(n, d=1/fs)  # Frequency bins

    # Only take the positive frequencies
    f = f[:n//2]
    cospec = cospec[:n//2]

    return f, cospec

"""
# --- TEST WITH SIMULATED DATA ---
# Simulate some data for cospectral analysis (vertical wind and temperature)
T = np.random.normal(25, 0.3, n)  # Simulated temperature signal

# Compute the cospectral density between vertical wind (w) and temperature (T)
frequencies, cospec = compute_cospectral_density(w, T, fs)

# Plot the cospectrum
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, np.abs(cospec))  # Absolute value of the cospectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Cospectral Density')
plt.title('Cospectrum Between Vertical Wind and Temperature')
plt.grid(True)
plt.show()
"""