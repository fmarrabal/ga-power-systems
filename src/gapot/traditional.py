"""
Traditional Power Theory Implementations
========================================

Implementation of IEEE 1459 and Instantaneous Power (p-q) Theory
for comparison with GAPoT methods.

These implementations follow the standard definitions to enable
direct comparison of results.

References
----------
.. [1] IEEE Std 1459-2010. IEEE Standard Definitions for the 
       Measurement of Electric Power Quantities.
.. [2] Akagi, H., Watanabe, E.H., Aredes, M. (2017). Instantaneous 
       Power Theory and Applications to Power Conditioning.
.. [3] Czarnecki, L.S. (2005). Currents' Physical Components (CPC).
"""

import numpy as np
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class IEEE1459Components:
    """Container for IEEE 1459 power components."""
    # Fundamental components
    P1: float      # Fundamental active power
    Q1: float      # Fundamental reactive power
    S1: float      # Fundamental apparent power
    PF1: float     # Fundamental (displacement) power factor
    
    # Harmonic components
    PH: float      # Harmonic active power
    DI: float      # Current distortion power
    DV: float      # Voltage distortion power
    SH: float      # Harmonic apparent power
    
    # Total
    P: float       # Total active power
    S: float       # Total apparent power
    N: float       # Non-active power
    D: float       # Distortion power (simplified)
    PF: float      # True power factor


class IEEE1459Power:
    """
    IEEE Standard 1459-2010 Power Calculations.
    
    Implements power definitions for single-phase systems with
    harmonics according to IEEE Std 1459.
    
    Parameters
    ----------
    u : np.ndarray
        Voltage signal samples.
    i : np.ndarray
        Current signal samples.
    f1 : float
        Fundamental frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to consider.
        
    Example
    -------
    >>> ieee = IEEE1459Power(u, i, f1=50, fs=50000)
    >>> print(f"P={ieee.P:.1f}W, Q1={ieee.Q1:.1f}var, PF={ieee.PF:.3f}")
    """
    
    def __init__(self, u: np.ndarray, i: np.ndarray, f1: float, fs: float,
                 n_harmonics: int = 50):
        self.u = u
        self.i = i
        self.f1 = f1
        self.fs = fs
        self.n_harmonics = n_harmonics
        
        self._calculate()
    
    def _calculate(self):
        """Calculate all IEEE 1459 power quantities."""
        # RMS values
        self.U_rms = np.sqrt(np.mean(self.u**2))
        self.I_rms = np.sqrt(np.mean(self.i**2))
        
        # FFT analysis
        N = len(self.u)
        U_fft = fft(self.u)
        I_fft = fft(self.i)
        freqs = fftfreq(N, 1/self.fs)
        
        # Extract harmonic components
        self.U_h = {}  # Voltage harmonics (complex)
        self.I_h = {}  # Current harmonics (complex)
        
        for h in range(self.n_harmonics + 1):
            f_h = h * self.f1
            idx = np.argmin(np.abs(freqs - f_h))
            
            if h == 0:  # DC
                self.U_h[h] = np.real(U_fft[idx]) / N
                self.I_h[h] = np.real(I_fft[idx]) / N
            else:
                self.U_h[h] = 2 * U_fft[idx] / N  # Complex phasor
                self.I_h[h] = 2 * I_fft[idx] / N
        
        # Fundamental quantities (h=1)
        U1 = np.abs(self.U_h[1]) / np.sqrt(2)  # RMS
        I1 = np.abs(self.I_h[1]) / np.sqrt(2)
        phi1 = np.angle(self.U_h[1]) - np.angle(self.I_h[1])
        
        self.U1 = U1
        self.I1 = I1
        self.phi1 = phi1
        
        # Fundamental powers
        self.P1 = U1 * I1 * np.cos(phi1)
        self.Q1 = U1 * I1 * np.sin(phi1)
        self.S1 = U1 * I1
        self.PF1 = np.cos(phi1)  # Displacement power factor
        
        # Harmonic RMS values
        U_H_sq = sum(np.abs(self.U_h[h])**2 / 2 for h in range(2, self.n_harmonics + 1)
                     if h in self.U_h)
        I_H_sq = sum(np.abs(self.I_h[h])**2 / 2 for h in range(2, self.n_harmonics + 1)
                     if h in self.I_h)
        
        self.U_H = np.sqrt(U_H_sq)
        self.I_H = np.sqrt(I_H_sq)
        
        # THD
        self.THD_U = self.U_H / U1 if U1 > 0 else 0
        self.THD_I = self.I_H / I1 if I1 > 0 else 0
        
        # Harmonic active power
        self.PH = sum(
            np.abs(self.U_h[h]) * np.abs(self.I_h[h]) / 2 * 
            np.cos(np.angle(self.U_h[h]) - np.angle(self.I_h[h]))
            for h in range(2, self.n_harmonics + 1)
            if h in self.U_h and h in self.I_h
        )
        
        # Total active power
        self.P = self.P1 + self.PH
        
        # Apparent power
        self.S = self.U_rms * self.I_rms
        
        # Non-active power
        self.N = np.sqrt(self.S**2 - self.P**2) if self.S**2 > self.P**2 else 0
        
        # Distortion powers (IEEE 1459 definitions)
        self.DI = self.U1 * self.I_H  # Current distortion
        self.DV = self.U_H * self.I1  # Voltage distortion
        self.SH = self.U_H * self.I_H  # Harmonic apparent
        
        # Simplified distortion power
        self.D = np.sqrt(self.S**2 - self.P**2 - self.Q1**2) if \
                 self.S**2 > self.P**2 + self.Q1**2 else 0
        
        # True power factor
        self.PF = self.P / self.S if self.S > 0 else 1.0
    
    def get_components(self) -> IEEE1459Components:
        """Get all power components as structured object."""
        return IEEE1459Components(
            P1=self.P1, Q1=self.Q1, S1=self.S1, PF1=self.PF1,
            PH=self.PH, DI=self.DI, DV=self.DV, SH=self.SH,
            P=self.P, S=self.S, N=self.N, D=self.D, PF=self.PF
        )
    
    def get_harmonic_powers(self) -> Dict[int, Dict[str, float]]:
        """Get power by harmonic number."""
        powers = {}
        for h in range(self.n_harmonics + 1):
            if h not in self.U_h or h not in self.I_h:
                continue
            
            U_h = np.abs(self.U_h[h]) / np.sqrt(2) if h > 0 else np.abs(self.U_h[h])
            I_h = np.abs(self.I_h[h]) / np.sqrt(2) if h > 0 else np.abs(self.I_h[h])
            phi_h = np.angle(self.U_h[h]) - np.angle(self.I_h[h])
            
            powers[h] = {
                'P': U_h * I_h * np.cos(phi_h),
                'Q': U_h * I_h * np.sin(phi_h),
                'S': U_h * I_h
            }
        
        return powers
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "IEEE 1459-2010 POWER ANALYSIS",
            "=" * 60,
            f"Fundamental frequency: {self.f1} Hz",
            "-" * 60,
            "RMS VALUES:",
            f"  U_rms = {self.U_rms:.4f} V",
            f"  I_rms = {self.I_rms:.4f} A",
            f"  U1 = {self.U1:.4f} V (fundamental)",
            f"  I1 = {self.I1:.4f} A (fundamental)",
            "-" * 60,
            "FUNDAMENTAL QUANTITIES:",
            f"  P1 = {self.P1:.4f} W",
            f"  Q1 = {self.Q1:.4f} var",
            f"  S1 = {self.S1:.4f} VA",
            f"  PF1 (displacement) = {self.PF1:.6f}",
            "-" * 60,
            "TOTAL QUANTITIES:",
            f"  P = {self.P:.4f} W",
            f"  S = {self.S:.4f} VA",
            f"  N = {self.N:.4f} va (non-active)",
            f"  D = {self.D:.4f} va (distortion)",
            f"  PF (true) = {self.PF:.6f}",
            "-" * 60,
            "DISTORTION:",
            f"  THD_U = {self.THD_U*100:.2f}%",
            f"  THD_I = {self.THD_I*100:.2f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class PQComponents:
    """Container for p-q theory components."""
    p: np.ndarray      # Instantaneous active power
    q: np.ndarray      # Instantaneous reactive power
    p_avg: float       # Average active power
    q_avg: float       # Average reactive power
    p_osc: np.ndarray  # Oscillating active power
    q_osc: np.ndarray  # Oscillating reactive power


class PQTheory:
    """
    Instantaneous Power Theory (p-q Theory) by Akagi.
    
    Originally developed for three-phase systems but adapted here
    for single-phase using Hilbert transform to create orthogonal signal.
    
    Parameters
    ----------
    u : np.ndarray
        Voltage signal samples.
    i : np.ndarray
        Current signal samples.
    fs : float
        Sampling frequency in Hz.
        
    Attributes
    ----------
    p : np.ndarray
        Instantaneous active power.
    q : np.ndarray
        Instantaneous reactive power.
    p_avg : float
        Average active power.
    q_avg : float
        Average reactive power.
        
    Example
    -------
    >>> pq = PQTheory(u, i, fs=50000)
    >>> print(f"P={pq.p_avg:.1f}W, Q={pq.q_avg:.1f}var")
    
    Notes
    -----
    For single-phase, the Hilbert transform creates the required
    orthogonal signal:
    u_alpha = u(t), u_beta = H{u(t)}
    where H{} is the Hilbert transform.
    """
    
    def __init__(self, u: np.ndarray, i: np.ndarray, fs: float):
        self.u = u
        self.i = i
        self.fs = fs
        
        self._calculate()
    
    def _calculate(self):
        """Calculate instantaneous powers using p-q theory."""
        from scipy.signal import hilbert
        
        # Create orthogonal components using Hilbert transform
        u_analytic = hilbert(self.u)
        i_analytic = hilbert(self.i)
        
        # Alpha-beta components
        u_alpha = self.u
        u_beta = np.imag(u_analytic)
        i_alpha = self.i
        i_beta = np.imag(i_analytic)
        
        # Instantaneous powers (standard p-q formulas)
        # p = u_α * i_α + u_β * i_β
        # q = u_β * i_α - u_α * i_β
        self.p = u_alpha * i_alpha + u_beta * i_beta
        self.q = u_beta * i_alpha - u_alpha * i_beta
        
        # Average values
        self.p_avg = np.mean(self.p)
        self.q_avg = np.mean(self.q)
        
        # Oscillating components
        self.p_osc = self.p - self.p_avg
        self.q_osc = self.q - self.q_avg
        
        # Additional metrics
        self.p_rms = np.sqrt(np.mean(self.p**2))
        self.q_rms = np.sqrt(np.mean(self.q**2))
    
    def get_components(self) -> PQComponents:
        """Get all p-q components."""
        return PQComponents(
            p=self.p, q=self.q,
            p_avg=self.p_avg, q_avg=self.q_avg,
            p_osc=self.p_osc, q_osc=self.q_osc
        )
    
    def get_current_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose current into active and reactive components.
        
        Returns
        -------
        i_p : np.ndarray
            Active current component.
        i_q : np.ndarray
            Reactive current component.
        """
        from scipy.signal import hilbert
        
        u_analytic = hilbert(self.u)
        u_alpha = self.u
        u_beta = np.imag(u_analytic)
        
        # Voltage magnitude squared
        u_sq = u_alpha**2 + u_beta**2
        
        # Current decomposition
        # i_p = p_avg * u / ||u||²
        # i_q comes from q component
        
        i_p_alpha = self.p_avg * u_alpha / u_sq
        i_p_beta = self.p_avg * u_beta / u_sq
        
        i_q_alpha = self.q_avg * u_beta / u_sq
        i_q_beta = -self.q_avg * u_alpha / u_sq
        
        # Return alpha components (original frame)
        return i_p_alpha, i_q_alpha
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "INSTANTANEOUS POWER THEORY (p-q) ANALYSIS",
            "=" * 60,
            "AVERAGE POWERS:",
            f"  p_avg = {self.p_avg:.4f} W",
            f"  q_avg = {self.q_avg:.4f} var",
            "-" * 60,
            "INSTANTANEOUS POWER STATISTICS:",
            f"  p_rms = {self.p_rms:.4f} W",
            f"  q_rms = {self.q_rms:.4f} var",
            f"  p_osc_rms = {np.sqrt(np.mean(self.p_osc**2)):.4f} W",
            f"  q_osc_rms = {np.sqrt(np.mean(self.q_osc**2)):.4f} var",
            "=" * 60,
        ]
        return "\n".join(lines)


class CPCTheory:
    """
    Currents' Physical Components (CPC) Theory by Czarnecki.
    
    Decomposes current into physically meaningful components:
    - Active current (power transfer)
    - Reactive current (energy oscillation)
    - Scattered current (load variation)
    - Unbalanced current (asymmetry)
    
    Parameters
    ----------
    u : np.ndarray
        Voltage signal.
    i : np.ndarray
        Current signal.
    f1 : float
        Fundamental frequency.
    fs : float
        Sampling frequency.
    """
    
    def __init__(self, u: np.ndarray, i: np.ndarray, f1: float, fs: float):
        self.u = u
        self.i = i
        self.f1 = f1
        self.fs = fs
        
        self._calculate()
    
    def _calculate(self):
        """Calculate CPC current components."""
        # FFT analysis
        N = len(self.u)
        U_fft = fft(self.u)
        I_fft = fft(self.i)
        freqs = fftfreq(N, 1/self.fs)
        
        # RMS values
        U_rms_sq = np.mean(self.u**2)
        
        # Active power and equivalent conductance
        self.P = np.mean(self.u * self.i)
        self.Ge = self.P / U_rms_sq if U_rms_sq > 0 else 0
        
        # Active current
        self.i_a = self.Ge * self.u
        
        # Reactive current (remaining)
        self.i_r = self.i - self.i_a
        
        # Norms
        self.I_a = np.sqrt(np.mean(self.i_a**2))
        self.I_r = np.sqrt(np.mean(self.i_r**2))
        self.I = np.sqrt(np.mean(self.i**2))
        
        # Verification: ||i||² = ||i_a||² + ||i_r||²
        self.orthogonality_error = abs(self.I**2 - self.I_a**2 - self.I_r**2)
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "CURRENTS' PHYSICAL COMPONENTS (CPC) ANALYSIS",
            "=" * 60,
            f"Active Power P = {self.P:.4f} W",
            f"Equivalent Conductance Ge = {self.Ge:.6f} S",
            "-" * 60,
            "CURRENT COMPONENTS:",
            f"  ||i|| = {self.I:.6f} A (total)",
            f"  ||i_a|| = {self.I_a:.6f} A (active)",
            f"  ||i_r|| = {self.I_r:.6f} A (reactive)",
            "-" * 60,
            f"Orthogonality error: {self.orthogonality_error:.2e}",
            "=" * 60,
        ]
        return "\n".join(lines)


def compare_methods(u: np.ndarray, i: np.ndarray, f1: float, fs: float) -> Dict:
    """
    Compare GAPoT, IEEE 1459, and p-q theory results.
    
    Returns dictionary with comparison table data.
    """
    from .power import GeometricPower
    
    gapot = GeometricPower(u, i, f1, fs)
    ieee = IEEE1459Power(u, i, f1, fs)
    pq = PQTheory(u, i, fs)
    
    return {
        'Method': ['GAPoT', 'IEEE 1459', 'p-q Theory'],
        'P (W)': [gapot.P, ieee.P, pq.p_avg],
        'Q/M_Q (var)': [gapot.M_Q_norm, ieee.Q1, pq.q_avg],
        'D/M_D': [gapot.M_D_norm, ieee.D, 'N/A'],
        'S (VA)': [gapot.S, ieee.S, 'N/A'],
        'PF': [gapot.PF, ieee.PF, 'N/A'],
    }
