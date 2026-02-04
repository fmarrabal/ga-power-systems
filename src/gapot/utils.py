"""
Utility Functions
=================

Helper functions for signal generation, analysis, and visualization.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


def generate_distorted_signal(
    U1: float = 230.0,
    I1: float = 10.0,
    phi1: float = 0.5,
    harmonics: Optional[List[Tuple[int, float, float, float]]] = None,
    f1: float = 50.0,
    fs: float = 10000.0,
    cycles: int = 2,
    dc_u: float = 0.0,
    dc_i: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate distorted voltage and current signals.
    
    Parameters
    ----------
    U1 : float
        Fundamental voltage RMS (V).
    I1 : float
        Fundamental current RMS (A).
    phi1 : float
        Fundamental phase shift (current lagging voltage) in radians.
    harmonics : list of tuples, optional
        List of (h, U_h_ratio, I_h_ratio, phi_h) where:
        - h: harmonic number
        - U_h_ratio: U_h/U1 ratio
        - I_h_ratio: I_h/I1 ratio  
        - phi_h: phase angle of h-th harmonic current
    f1 : float
        Fundamental frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    cycles : int
        Number of fundamental cycles.
    dc_u : float
        DC offset in voltage.
    dc_i : float
        DC offset in current.
        
    Returns
    -------
    u : np.ndarray
        Voltage signal.
    i : np.ndarray
        Current signal.
    t : np.ndarray
        Time vector.
        
    Example
    -------
    >>> u, i, t = generate_distorted_signal(
    ...     U1=230, I1=10, phi1=0.5,
    ...     harmonics=[(5, 0.05, 0.20, 0.3), (7, 0.03, 0.14, 0.2)],
    ...     f1=50, fs=50000, cycles=2
    ... )
    """
    T1 = 1 / f1  # Fundamental period
    duration = cycles * T1
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    omega1 = 2 * np.pi * f1
    
    # Fundamental
    u = dc_u + U1 * np.sqrt(2) * np.sin(omega1 * t)
    i = dc_i + I1 * np.sqrt(2) * np.sin(omega1 * t - phi1)
    
    # Add harmonics
    if harmonics:
        for h, U_h_ratio, I_h_ratio, phi_h in harmonics:
            U_h = U1 * U_h_ratio
            I_h = I1 * I_h_ratio
            omega_h = h * omega1
            
            u += U_h * np.sqrt(2) * np.sin(omega_h * t)
            i += I_h * np.sqrt(2) * np.sin(omega_h * t - phi_h)
    
    return u, i, t


def generate_interharmonic_signal(
    U1: float = 230.0,
    I1: float = 10.0,
    phi1: float = 0.5,
    interharmonics: Optional[List[Tuple[float, float, float, float]]] = None,
    f1: float = 50.0,
    fs: float = 10000.0,
    cycles: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate signal with interharmonics (non-integer frequency ratios).
    
    Parameters
    ----------
    interharmonics : list of tuples
        List of (f_ih, U_ih_ratio, I_ih_ratio, phi_ih) where f_ih is
        the interharmonic frequency in Hz.
        
    Returns
    -------
    u, i, t : np.ndarray
        Voltage, current, and time arrays.
    """
    T1 = 1 / f1
    duration = cycles * T1
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    omega1 = 2 * np.pi * f1
    
    # Fundamental
    u = U1 * np.sqrt(2) * np.sin(omega1 * t)
    i = I1 * np.sqrt(2) * np.sin(omega1 * t - phi1)
    
    # Add interharmonics
    if interharmonics:
        for f_ih, U_ih_ratio, I_ih_ratio, phi_ih in interharmonics:
            U_ih = U1 * U_ih_ratio
            I_ih = I1 * I_ih_ratio
            omega_ih = 2 * np.pi * f_ih
            
            u += U_ih * np.sqrt(2) * np.sin(omega_ih * t)
            i += I_ih * np.sqrt(2) * np.sin(omega_ih * t - phi_ih)
    
    return u, i, t


def add_noise(signal: np.ndarray, snr_db: float = 40.0) -> np.ndarray:
    """
    Add Gaussian noise to signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Clean signal.
    snr_db : float
        Signal-to-noise ratio in dB.
        
    Returns
    -------
    noisy : np.ndarray
        Signal with added noise.
    """
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


def rms(signal: np.ndarray) -> float:
    """Calculate RMS value of signal."""
    return np.sqrt(np.mean(signal**2))


def thd(signal: np.ndarray, f1: float, fs: float, n_harmonics: int = 50) -> float:
    """
    Calculate Total Harmonic Distortion (THD).
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    f1 : float
        Fundamental frequency.
    fs : float
        Sampling frequency.
    n_harmonics : int
        Number of harmonics to include.
        
    Returns
    -------
    thd : float
        THD as a ratio (not percentage).
    """
    from scipy.fft import fft, fftfreq
    
    N = len(signal)
    X = fft(signal)
    freqs = fftfreq(N, 1/fs)
    
    # Find fundamental
    idx1 = np.argmin(np.abs(freqs - f1))
    X1 = np.abs(X[idx1])
    
    # Sum harmonic magnitudes squared
    X_h_sq_sum = 0
    for h in range(2, n_harmonics + 1):
        f_h = h * f1
        if f_h > fs / 2:
            break
        idx_h = np.argmin(np.abs(freqs - f_h))
        X_h_sq_sum += np.abs(X[idx_h])**2
    
    if X1 > 0:
        return np.sqrt(X_h_sq_sum) / X1
    else:
        return 0.0


def power_factor_from_signals(u: np.ndarray, i: np.ndarray) -> float:
    """
    Calculate power factor from time-domain signals.
    
    PF = P / S = <u*i> / (U_rms * I_rms)
    """
    P = np.mean(u * i)
    S = rms(u) * rms(i)
    return P / S if S > 0 else 1.0


def displacement_factor(u: np.ndarray, i: np.ndarray, f1: float, fs: float) -> float:
    """
    Calculate displacement power factor (fundamental only).
    """
    from scipy.fft import fft, fftfreq
    
    N = len(u)
    U_fft = fft(u)
    I_fft = fft(i)
    freqs = fftfreq(N, 1/fs)
    
    idx1 = np.argmin(np.abs(freqs - f1))
    
    phi1 = np.angle(U_fft[idx1]) - np.angle(I_fft[idx1])
    
    return np.cos(phi1)


def generate_three_phase(
    U: float = 230.0,
    I: float = 10.0,
    phi: float = 0.5,
    f1: float = 50.0,
    fs: float = 10000.0,
    cycles: int = 2,
    unbalance: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate three-phase voltage and current signals.
    
    Parameters
    ----------
    U : float
        Phase voltage RMS.
    I : float
        Phase current RMS.
    phi : float
        Power factor angle.
    f1 : float
        Fundamental frequency.
    fs : float
        Sampling frequency.
    cycles : int
        Number of cycles.
    unbalance : float
        Voltage unbalance ratio (0 to 1).
        
    Returns
    -------
    u_a, u_b, u_c : np.ndarray
        Three-phase voltages.
    i_a, i_b, i_c : np.ndarray
        Three-phase currents.
    t : np.ndarray
        Time vector.
    """
    T1 = 1 / f1
    duration = cycles * T1
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    omega = 2 * np.pi * f1
    
    # Phase shifts (120° apart)
    phase_a = 0
    phase_b = -2 * np.pi / 3
    phase_c = 2 * np.pi / 3
    
    # Voltages (with optional unbalance)
    u_a = U * np.sqrt(2) * (1 + unbalance) * np.sin(omega * t + phase_a)
    u_b = U * np.sqrt(2) * (1 - 0.5*unbalance) * np.sin(omega * t + phase_b)
    u_c = U * np.sqrt(2) * (1 - 0.5*unbalance) * np.sin(omega * t + phase_c)
    
    # Currents (lagging by phi)
    i_a = I * np.sqrt(2) * np.sin(omega * t + phase_a - phi)
    i_b = I * np.sqrt(2) * np.sin(omega * t + phase_b - phi)
    i_c = I * np.sqrt(2) * np.sin(omega * t + phase_c - phi)
    
    return u_a, u_b, u_c, i_a, i_b, i_c, t


def create_comparison_table(
    u: np.ndarray, 
    i: np.ndarray, 
    f1: float, 
    fs: float
) -> str:
    """
    Create formatted comparison table of different methods.
    
    Returns ASCII table string.
    """
    from .power import GeometricPower
    from .traditional import IEEE1459Power, PQTheory
    
    gapot = GeometricPower(u, i, f1, fs)
    ieee = IEEE1459Power(u, i, f1, fs)
    pq = PQTheory(u, i, fs)
    
    lines = [
        "=" * 70,
        "POWER CALCULATION COMPARISON",
        "=" * 70,
        f"{'Method':<15} {'P (W)':<12} {'Q (var)':<12} {'D (va)':<12} {'S (VA)':<12} {'PF':<8}",
        "-" * 70,
        f"{'GAPoT':<15} {gapot.P:<12.4f} {gapot.M_Q_norm:<12.4f} {gapot.M_D_norm:<12.4f} {gapot.S:<12.4f} {gapot.PF:<8.4f}",
        f"{'IEEE 1459':<15} {ieee.P:<12.4f} {ieee.Q1:<12.4f} {ieee.D:<12.4f} {ieee.S:<12.4f} {ieee.PF:<8.4f}",
        f"{'p-q Theory':<15} {pq.p_avg:<12.4f} {pq.q_avg:<12.4f} {'N/A':<12} {'N/A':<12} {'N/A':<8}",
        "=" * 70,
    ]
    
    return "\n".join(lines)
