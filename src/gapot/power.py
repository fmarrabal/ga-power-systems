"""
Geometric Power Calculation Module
===================================

Implementation of Geometric Algebra Power Theory (GAPoT) for
calculating power components in electrical systems.

References
----------
.. [1] Menti et al. (2007). Geometric algebra: A powerful tool for 
       representing power under nonsinusoidal conditions.
.. [2] Castro-Núñez & Castro-Puche (2012). The IEEE Standard 1459, 
       the CPC Power Theory, and Geometric Algebra.
.. [3] Montoya et al. (2021). Vector Geometric Algebra in Power Systems.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from .basis import FourierToGA, GABasis, SpectralComponent


@dataclass
class PowerComponents:
    """Container for geometric power components."""
    P: float  # Active power (scalar)
    M_Q: np.ndarray  # Reactive power (bivector coefficients)
    M_D: np.ndarray  # Distortion power (cross-frequency bivector)
    M_Q_norm: float  # ||M_Q||
    M_D_norm: float  # ||M_D||
    S: float  # Apparent power ||M||
    PF: float  # Power factor P/S


class GeometricPower:
    """
    Geometric Power Calculator using GAPoT.
    
    Calculates the geometric power M = ui as the geometric product
    of voltage and current vectors, decomposed into:
    - P: Active power (scalar part)
    - M_Q: Reactive power (same-frequency bivector)
    - M_D: Distortion power (cross-frequency bivector)
    
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
    threshold : float, optional
        FFT magnitude threshold. Default 0.01.
        
    Attributes
    ----------
    P : float
        Active power in Watts.
    M_Q_norm : float
        Reactive power magnitude in var.
    M_D_norm : float
        Distortion power magnitude.
    S : float
        Apparent power in VA.
    PF : float
        Geometric power factor.
    u_vec : np.ndarray
        Voltage GA vector.
    i_vec : np.ndarray
        Current GA vector.
        
    Example
    -------
    >>> import numpy as np
    >>> t = np.linspace(0, 0.04, 2000)
    >>> u = 230*np.sqrt(2) * np.sin(2*np.pi*50*t)
    >>> i = 10*np.sqrt(2) * np.sin(2*np.pi*50*t - 0.5)
    >>> gp = GeometricPower(u, i, f1=50, fs=50000)
    >>> print(f"P={gp.P:.1f}W, Q={gp.M_Q_norm:.1f}var, PF={gp.PF:.3f}")
    """
    
    def __init__(self, u: np.ndarray, i: np.ndarray, f1: float, fs: float,
                 threshold: float = 0.01):
        self.u_signal = u
        self.i_signal = i
        self.f1 = f1
        self.fs = fs
        self.threshold = threshold
        
        # Transform signals to GA
        self._transform_signals()
        
        # Calculate power components
        self._calculate_power()
    
    def _transform_signals(self):
        """Transform voltage and current to GA vectors."""
        # Transform voltage
        self.u_converter = FourierToGA(self.fs, self.f1, self.threshold)
        self.u_vec, self.u_components = self.u_converter.transform(self.u_signal)
        
        # Transform current using same frequency set for consistency
        self.i_converter = FourierToGA(self.fs, self.f1, self.threshold)
        self.i_vec, self.i_components = self.i_converter.transform(self.i_signal)
        
        # Build unified frequency set
        u_freqs = set(c.frequency for c in self.u_components)
        i_freqs = set(c.frequency for c in self.i_components)
        self.all_freqs = sorted(u_freqs | i_freqs)
        
        # Rebuild vectors with unified basis
        self._unify_basis()
    
    def _unify_basis(self):
        """Ensure voltage and current use the same basis."""
        n_freqs = len(self.all_freqs)
        
        # Determine dimension
        has_dc = 0 in self.all_freqs
        n_ac = n_freqs - (1 if has_dc else 0)
        self.dimension = (1 if has_dc else 0) + 2 * n_ac
        
        # Create unified vectors
        self.u_unified = np.zeros(self.dimension)
        self.i_unified = np.zeros(self.dimension)
        
        # Map components to unified basis
        for comp in self.u_components:
            idx_cos, idx_sin = self._get_unified_indices(comp.frequency)
            if comp.is_dc:
                self.u_unified[idx_cos] = comp.cosine_coeff
            else:
                self.u_unified[idx_cos] = comp.cosine_coeff
                self.u_unified[idx_sin] = comp.sine_coeff
        
        for comp in self.i_components:
            idx_cos, idx_sin = self._get_unified_indices(comp.frequency)
            if comp.is_dc:
                self.i_unified[idx_cos] = comp.cosine_coeff
            else:
                self.i_unified[idx_cos] = comp.cosine_coeff
                self.i_unified[idx_sin] = comp.sine_coeff
        
        self.u_vec = self.u_unified
        self.i_vec = self.i_unified
    
    def _get_unified_indices(self, frequency: float) -> Tuple[int, Optional[int]]:
        """Get indices in unified basis for a frequency."""
        m = self.all_freqs.index(frequency)
        has_dc = 0 in self.all_freqs
        
        if frequency == 0:
            return (0, None)
        
        # AC component position
        ac_pos = m - (1 if has_dc else 0)
        offset = 1 if has_dc else 0
        
        return (offset + 2*ac_pos, offset + 2*ac_pos + 1)
    
    def _calculate_power(self):
        """Calculate geometric power components."""
        # Active power: P = Σ (U_hc*I_hc + U_hs*I_hs) / 2 for each harmonic
        # This is the scalar part of the geometric product
        
        self.P = 0.0
        self.M_Q_components = {}  # Same-frequency reactive
        self.M_D_components = {}  # Cross-frequency distortion
        
        has_dc = 0 in self.all_freqs
        
        # Process each frequency
        for freq in self.all_freqs:
            idx_cos, idx_sin = self._get_unified_indices(freq)
            
            U_c = self.u_vec[idx_cos]
            I_c = self.i_vec[idx_cos]
            
            if freq == 0:  # DC
                self.P += U_c * I_c
            else:
                U_s = self.u_vec[idx_sin]
                I_s = self.i_vec[idx_sin]
                
                # Active power contribution: (U_c*I_c + U_s*I_s) / 2
                self.P += 0.5 * (U_c * I_c + U_s * I_s)
                
                # Reactive power contribution: (U_c*I_s - U_s*I_c) / 2
                Q_h = 0.5 * (U_c * I_s - U_s * I_c)
                if abs(Q_h) > 1e-10:
                    self.M_Q_components[freq] = Q_h
        
        # Cross-frequency distortion power
        ac_freqs = [f for f in self.all_freqs if f > 0]
        
        for i, f_m in enumerate(ac_freqs):
            for j, f_n in enumerate(ac_freqs):
                if i >= j:  # Only upper triangle (m != n)
                    continue
                
                idx_m_c, idx_m_s = self._get_unified_indices(f_m)
                idx_n_c, idx_n_s = self._get_unified_indices(f_n)
                
                U_mc, U_ms = self.u_vec[idx_m_c], self.u_vec[idx_m_s]
                I_nc, I_ns = self.i_vec[idx_n_c], self.i_vec[idx_n_s]
                U_nc, U_ns = self.u_vec[idx_n_c], self.u_vec[idx_n_s]
                I_mc, I_ms = self.i_vec[idx_m_c], self.i_vec[idx_m_s]
                
                # Cross-frequency term: (U_mc*I_ns - U_ms*I_nc) σ_m ∧ σ_n
                # Plus: (U_nc*I_ms - U_ns*I_mc) σ_n ∧ σ_m = -(U_nc*I_ms - U_ns*I_mc) σ_m ∧ σ_n
                D_mn = 0.5 * ((U_mc*I_ns - U_ms*I_nc) - (U_nc*I_ms - U_ns*I_mc))
                
                if abs(D_mn) > 1e-10:
                    self.M_D_components[(f_m, f_n)] = D_mn
        
        # Calculate norms
        self.M_Q_norm = np.sqrt(sum(q**2 for q in self.M_Q_components.values()))
        self.M_D_norm = np.sqrt(sum(d**2 for d in self.M_D_components.values()))
        
        # Apparent power: ||M|| = sqrt(P² + ||M_Q||² + ||M_D||²)
        self.S = np.sqrt(self.P**2 + self.M_Q_norm**2 + self.M_D_norm**2)
        
        # Power factor
        self.PF = self.P / self.S if self.S > 0 else 1.0
        
        # Also calculate from RMS values for verification
        self.U_rms = np.sqrt(np.mean(self.u_signal**2))
        self.I_rms = np.sqrt(np.mean(self.i_signal**2))
        self.S_traditional = self.U_rms * self.I_rms
    
    def get_power_components(self) -> PowerComponents:
        """Get all power components as a structured object."""
        return PowerComponents(
            P=self.P,
            M_Q=np.array(list(self.M_Q_components.values())),
            M_D=np.array(list(self.M_D_components.values())),
            M_Q_norm=self.M_Q_norm,
            M_D_norm=self.M_D_norm,
            S=self.S,
            PF=self.PF
        )
    
    def get_harmonic_powers(self) -> Dict[float, Dict[str, float]]:
        """
        Get power decomposition by harmonic.
        
        Returns
        -------
        powers : dict
            Dictionary mapping frequency to {P_h, Q_h} values.
        """
        powers = {}
        
        for freq in self.all_freqs:
            idx_cos, idx_sin = self._get_unified_indices(freq)
            
            U_c = self.u_vec[idx_cos]
            I_c = self.i_vec[idx_cos]
            
            if freq == 0:
                powers[freq] = {'P': U_c * I_c, 'Q': 0.0}
            else:
                U_s = self.u_vec[idx_sin]
                I_s = self.i_vec[idx_sin]
                
                P_h = 0.5 * (U_c * I_c + U_s * I_s)
                Q_h = 0.5 * (U_c * I_s - U_s * I_c)
                
                powers[freq] = {'P': P_h, 'Q': Q_h}
        
        return powers
    
    def verify_energy_conservation(self) -> Dict[str, float]:
        """
        Verify energy conservation: ||M||² = S² = U_rms² × I_rms².
        
        Returns
        -------
        results : dict
            Verification results including relative error.
        """
        S_ga = self.S
        S_trad = self.S_traditional
        
        return {
            'S_geometric': S_ga,
            'S_traditional': S_trad,
            'relative_error': abs(S_ga - S_trad) / S_trad if S_trad > 0 else 0,
            'P': self.P,
            'M_Q_norm': self.M_Q_norm,
            'M_D_norm': self.M_D_norm,
            'P_squared_plus_nonactive': self.P**2 + self.M_Q_norm**2 + self.M_D_norm**2,
            'S_squared': S_ga**2
        }
    
    def summary(self) -> str:
        """Generate a text summary of power analysis."""
        lines = [
            "=" * 60,
            "GEOMETRIC POWER ANALYSIS SUMMARY",
            "=" * 60,
            f"Fundamental frequency: {self.f1} Hz",
            f"Number of harmonics: {len(self.all_freqs)}",
            "-" * 60,
            "POWER COMPONENTS:",
            f"  Active Power P        = {self.P:12.4f} W",
            f"  Reactive Power ||M_Q||= {self.M_Q_norm:12.4f} var",
            f"  Distortion Power ||M_D||= {self.M_D_norm:12.4f} va",
            f"  Apparent Power ||M||  = {self.S:12.4f} VA",
            "-" * 60,
            "POWER FACTOR:",
            f"  Geometric PF = P/||M|| = {self.PF:.6f}",
            "-" * 60,
            "VERIFICATION (Energy Conservation):",
            f"  U_rms = {self.U_rms:.4f} V",
            f"  I_rms = {self.I_rms:.4f} A",
            f"  S_traditional = U_rms × I_rms = {self.S_traditional:.4f} VA",
            f"  S_geometric = ||M|| = {self.S:.4f} VA",
            f"  Relative error = {abs(self.S - self.S_traditional)/self.S_traditional*100:.4f}%",
            "=" * 60,
        ]
        
        # Add harmonic breakdown if multiple harmonics
        if len(self.all_freqs) > 1:
            lines.append("HARMONIC BREAKDOWN:")
            harmonic_powers = self.get_harmonic_powers()
            for freq, pwr in sorted(harmonic_powers.items()):
                h = int(freq / self.f1) if self.f1 > 0 else 0
                label = "DC" if freq == 0 else f"h={h}" if freq % self.f1 == 0 else f"f={freq}Hz"
                lines.append(f"  {label:8s}: P={pwr['P']:10.4f} W, Q={pwr['Q']:10.4f} var")
            lines.append("=" * 60)
        
        # Add distortion breakdown if present
        if self.M_D_components:
            lines.append("CROSS-FREQUENCY DISTORTION TERMS:")
            for (f_m, f_n), D_mn in sorted(self.M_D_components.items()):
                lines.append(f"  ({f_m:.0f}Hz × {f_n:.0f}Hz): D = {D_mn:.4f}")
            lines.append("=" * 60)
        
        return "\n".join(lines)


def calculate_geometric_power(u: np.ndarray, i: np.ndarray, 
                              f1: float = 50.0, fs: float = 10000.0) -> Dict[str, float]:
    """
    Convenience function for quick power calculation.
    
    Parameters
    ----------
    u : np.ndarray
        Voltage samples.
    i : np.ndarray
        Current samples.
    f1 : float
        Fundamental frequency (default 50 Hz).
    fs : float
        Sampling frequency (default 10 kHz).
        
    Returns
    -------
    results : dict
        Dictionary with P, Q, D, S, PF values.
    """
    gp = GeometricPower(u, i, f1, fs)
    return {
        'P': gp.P,
        'Q': gp.M_Q_norm,
        'D': gp.M_D_norm,
        'S': gp.S,
        'PF': gp.PF
    }
