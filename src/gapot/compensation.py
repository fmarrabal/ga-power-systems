"""
Current Decomposition and Compensation Module
==============================================

Implementation of geometric current decomposition for power
factor compensation using projection and rejection operations.

References
----------
.. [1] Montoya et al. (2020). A new approach to single-phase systems
       under sinusoidal and non-sinusoidal supply using geometric algebra.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CurrentComponents:
    """Container for decomposed current components."""
    i_active: np.ndarray      # Active current (collinear with voltage)
    i_reactive: np.ndarray    # Reactive current (same-frequency, quadrature)
    i_distortion: np.ndarray  # Distortion current (cross-frequency)
    i_total: np.ndarray       # Total current (sum)


class CurrentDecomposition:
    """
    Geometric Current Decomposition.
    
    Decomposes current into orthogonal components using GA projection
    and rejection operations:
    - i_P: Active current (projection onto voltage)
    - i_Q: Reactive current (same-frequency, orthogonal to voltage)
    - i_D: Distortion current (cross-frequency components)
    
    Parameters
    ----------
    u_vec : np.ndarray
        Voltage GA vector.
    i_vec : np.ndarray
        Current GA vector.
    frequency_info : dict, optional
        Frequency mapping information from FourierToGA.
        
    Attributes
    ----------
    i_active : np.ndarray
        Active current vector.
    i_reactive : np.ndarray
        Reactive current vector.
    i_distortion : np.ndarray
        Distortion current vector.
        
    Example
    -------
    >>> from gapot import GeometricPower, CurrentDecomposition
    >>> gp = GeometricPower(u, i, f1=50, fs=50000)
    >>> decomp = CurrentDecomposition(gp.u_vec, gp.i_vec)
    >>> print(f"Active current norm: {decomp.i_active_norm}")
    """
    
    def __init__(self, u_vec: np.ndarray, i_vec: np.ndarray,
                 frequency_info: Optional[Dict] = None):
        self.u_vec = u_vec
        self.i_vec = i_vec
        self.frequency_info = frequency_info
        
        self._decompose()
    
    def _decompose(self):
        """Perform current decomposition using geometric operations."""
        # Voltage norm squared
        u_norm_sq = np.dot(self.u_vec, self.u_vec)
        
        if u_norm_sq < 1e-10:
            # Zero voltage case
            self.i_active = np.zeros_like(self.i_vec)
            self.i_reactive = np.zeros_like(self.i_vec)
            self.i_distortion = self.i_vec.copy()
            return
        
        # Active current: projection of i onto u
        # i_P = (i · u) / ||u||² × u
        i_dot_u = np.dot(self.i_vec, self.u_vec)
        self.i_active = (i_dot_u / u_norm_sq) * self.u_vec
        
        # Non-active current
        i_nonactive = self.i_vec - self.i_active
        
        # For full decomposition into reactive and distortion,
        # we need frequency information
        if self.frequency_info is not None:
            self._decompose_nonactive(i_nonactive)
        else:
            # Simple decomposition: all non-active treated as reactive
            self.i_reactive = i_nonactive
            self.i_distortion = np.zeros_like(self.i_vec)
        
        # Calculate norms
        self.i_active_norm = np.linalg.norm(self.i_active)
        self.i_reactive_norm = np.linalg.norm(self.i_reactive)
        self.i_distortion_norm = np.linalg.norm(self.i_distortion)
        self.i_total_norm = np.linalg.norm(self.i_vec)
    
    def _decompose_nonactive(self, i_nonactive: np.ndarray):
        """Decompose non-active current into reactive and distortion."""
        # This requires frequency-wise analysis
        # For now, use simple approximation
        self.i_reactive = i_nonactive
        self.i_distortion = np.zeros_like(self.i_vec)
    
    def get_components(self) -> CurrentComponents:
        """Get all current components."""
        return CurrentComponents(
            i_active=self.i_active,
            i_reactive=self.i_reactive,
            i_distortion=self.i_distortion,
            i_total=self.i_vec
        )
    
    def get_compensation_current(self, 
                                  compensate_reactive: bool = True,
                                  compensate_distortion: bool = True) -> np.ndarray:
        """
        Calculate required compensation current.
        
        Parameters
        ----------
        compensate_reactive : bool
            Compensate reactive current component.
        compensate_distortion : bool
            Compensate distortion current component.
            
        Returns
        -------
        i_comp : np.ndarray
            Compensation current vector (inject negative of non-active).
        """
        i_comp = np.zeros_like(self.i_vec)
        
        if compensate_reactive:
            i_comp -= self.i_reactive
        
        if compensate_distortion:
            i_comp -= self.i_distortion
        
        return i_comp
    
    def get_compensated_current(self,
                                 compensate_reactive: bool = True,
                                 compensate_distortion: bool = True) -> np.ndarray:
        """
        Calculate supply current after compensation.
        
        Returns
        -------
        i_supply : np.ndarray
            Supply current after compensation (ideally just i_active).
        """
        i_comp = self.get_compensation_current(compensate_reactive, 
                                                compensate_distortion)
        return self.i_vec + i_comp
    
    def verify_orthogonality(self) -> Dict[str, float]:
        """Verify orthogonality of decomposed components."""
        return {
            'active_reactive': np.dot(self.i_active, self.i_reactive),
            'active_distortion': np.dot(self.i_active, self.i_distortion),
            'reactive_distortion': np.dot(self.i_reactive, self.i_distortion),
            'norm_check': (self.i_active_norm**2 + self.i_reactive_norm**2 + 
                          self.i_distortion_norm**2 - self.i_total_norm**2)
        }
    
    def summary(self) -> str:
        """Generate summary of current decomposition."""
        lines = [
            "=" * 50,
            "CURRENT DECOMPOSITION SUMMARY",
            "=" * 50,
            f"Total current ||i||     = {self.i_total_norm:.6f} A",
            f"Active current ||i_P||  = {self.i_active_norm:.6f} A",
            f"Reactive current ||i_Q||= {self.i_reactive_norm:.6f} A",
            f"Distortion current ||i_D||= {self.i_distortion_norm:.6f} A",
            "-" * 50,
            "ORTHOGONALITY CHECK:",
        ]
        
        orth = self.verify_orthogonality()
        lines.append(f"  i_P · i_Q = {orth['active_reactive']:.2e}")
        lines.append(f"  i_P · i_D = {orth['active_distortion']:.2e}")
        lines.append(f"  i_Q · i_D = {orth['reactive_distortion']:.2e}")
        lines.append(f"  ||i||² - Σ||i_k||² = {orth['norm_check']:.2e}")
        lines.append("=" * 50)
        
        return "\n".join(lines)


def project_onto(vector: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """
    Project vector onto another vector.
    
    proj_b(a) = (a · b / ||b||²) b
    """
    onto_norm_sq = np.dot(onto, onto)
    if onto_norm_sq < 1e-10:
        return np.zeros_like(vector)
    return (np.dot(vector, onto) / onto_norm_sq) * onto


def reject_from(vector: np.ndarray, from_vec: np.ndarray) -> np.ndarray:
    """
    Reject vector from another vector (orthogonal complement).
    
    rej_b(a) = a - proj_b(a)
    """
    return vector - project_onto(vector, from_vec)
