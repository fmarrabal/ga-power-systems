"""
Fourier-to-GA Basis Mapping Module
==================================

Implementation of Algorithm 1 from the paper: Fourier-to-GA Transformation.

This module provides the mapping between Fourier frequency domain 
and geometric algebra vector space, including handling of DC, harmonics,
and interharmonics.

References
----------
.. [1] Montoya et al. (2026). Section on Fourier-to-GA Basis Mapping
       and Indexing Strategy.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

try:
    from clifford import Cl
except ImportError:
    Cl = None
    print("Warning: clifford package not installed. Using numpy-based implementation.")


@dataclass
class SpectralComponent:
    """Represents a single spectral component."""
    frequency: float
    magnitude: float
    phase: float
    cosine_coeff: float
    sine_coeff: float
    basis_index_cos: int
    basis_index_sin: Optional[int]
    is_dc: bool = False


class GABasis:
    """
    Geometric Algebra basis for power systems.
    
    Creates an orthonormal basis for representing voltage and current
    as geometric vectors in a Euclidean space G_n.
    
    Parameters
    ----------
    n_components : int
        Number of frequency components (including DC).
    use_clifford : bool, optional
        Use clifford library for symbolic computation. Default True.
        
    Attributes
    ----------
    dimension : int
        Dimension of the vector space (2*N for N AC components + 1 for DC).
    basis_vectors : list
        List of basis vectors σ_0, σ_1, ..., σ_n.
        
    Example
    -------
    >>> basis = GABasis(n_components=3)  # DC + 2 harmonics
    >>> print(f"Dimension: {basis.dimension}")  # 5
    """
    
    def __init__(self, n_components: int, use_clifford: bool = True):
        self.n_components = n_components
        self.use_clifford = use_clifford and (Cl is not None)
        
        # Dimension: 1 for DC + 2 for each AC component
        self.dimension = 1 + 2 * (n_components - 1) if n_components > 1 else 1
        
        if self.use_clifford:
            self._init_clifford_basis()
        else:
            self._init_numpy_basis()
    
    def _init_clifford_basis(self):
        """Initialize using clifford library."""
        layout, blades = Cl(self.dimension)
        self.layout = layout
        self.blades = blades
        
        # Extract basis vectors
        self.basis_vectors = []
        for i in range(self.dimension):
            self.basis_vectors.append(blades[f'e{i+1}'])
    
    def _init_numpy_basis(self):
        """Initialize using numpy (orthonormal vectors)."""
        self.layout = None
        self.blades = None
        self.basis_vectors = []
        
        for i in range(self.dimension):
            e_i = np.zeros(self.dimension)
            e_i[i] = 1.0
            self.basis_vectors.append(e_i)
    
    def get_basis_vector(self, index: int):
        """Get basis vector by index."""
        if index < 0 or index >= self.dimension:
            raise IndexError(f"Basis index {index} out of range [0, {self.dimension-1}]")
        return self.basis_vectors[index]
    
    def inner_product(self, v1, v2) -> float:
        """Compute inner product of two vectors."""
        if self.use_clifford:
            return float((v1 | v2).value)
        else:
            return float(np.dot(v1, v2))
    
    def outer_product(self, v1, v2):
        """Compute outer (wedge) product of two vectors."""
        if self.use_clifford:
            return v1 ^ v2
        else:
            # Return antisymmetric tensor representation
            return np.outer(v1, v2) - np.outer(v2, v1)
    
    def geometric_product(self, v1, v2):
        """Compute geometric product of two vectors."""
        if self.use_clifford:
            return v1 * v2
        else:
            # For vectors: ab = a·b + a∧b
            inner = self.inner_product(v1, v2)
            outer = self.outer_product(v1, v2)
            return {'scalar': inner, 'bivector': outer}
    
    def norm(self, v) -> float:
        """Compute norm of a vector."""
        if self.use_clifford:
            return float(abs(v))
        else:
            return float(np.linalg.norm(v))


class FourierToGA:
    """
    Fourier-to-GA Transformation (Algorithm 1).
    
    Transforms time-domain signals to their geometric algebra representation
    following the indexing strategy defined in the paper.
    
    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    f1 : float
        Fundamental frequency in Hz.
    threshold : float, optional
        Magnitude threshold for significant spectral components.
        Components below this fraction of max are ignored. Default 0.01.
    max_harmonics : int, optional
        Maximum number of harmonics to consider. Default 50.
        
    Attributes
    ----------
    frequency_set : list
        Ordered set of detected frequencies.
    basis : GABasis
        The geometric algebra basis.
        
    Example
    -------
    >>> import numpy as np
    >>> t = np.linspace(0, 0.04, 2000)
    >>> u = 230*np.sqrt(2) * np.sin(2*np.pi*50*t)
    >>> converter = FourierToGA(fs=50000, f1=50)
    >>> u_vec, components = converter.transform(u)
    """
    
    def __init__(self, fs: float, f1: float, threshold: float = 0.01,
                 max_harmonics: int = 50):
        self.fs = fs
        self.f1 = f1
        self.threshold = threshold
        self.max_harmonics = max_harmonics
        
        self.frequency_set: List[float] = []
        self.components: List[SpectralComponent] = []
        self.basis: Optional[GABasis] = None
        
    def transform(self, signal: np.ndarray) -> Tuple[np.ndarray, List[SpectralComponent]]:
        """
        Transform time-domain signal to GA vector.
        
        Implements Algorithm 1 from the paper:
        1. Spectral analysis via FFT
        2. Coefficient extraction
        3. Basis assignment
        4. Vector construction
        
        Parameters
        ----------
        signal : np.ndarray
            Time-domain signal samples.
            
        Returns
        -------
        ga_vector : np.ndarray
            Geometric algebra vector representation.
        components : list of SpectralComponent
            Detected spectral components with metadata.
        """
        N = len(signal)
        
        # Step 1: Spectral Analysis
        X = fft(signal)
        freqs = fftfreq(N, 1/self.fs)
        
        # Only positive frequencies
        pos_mask = freqs >= 0
        X_pos = X[pos_mask]
        freqs_pos = freqs[pos_mask]
        
        # Find significant components
        magnitudes = np.abs(X_pos) / N
        magnitudes[1:] *= 2  # Account for negative frequencies
        
        max_mag = np.max(magnitudes)
        significant = magnitudes > self.threshold * max_mag
        
        # Step 2: Coefficient Extraction
        self.components = []
        self.frequency_set = []
        
        for idx in np.where(significant)[0]:
            freq = freqs_pos[idx]
            
            # Skip frequencies beyond max harmonics
            if freq > self.max_harmonics * self.f1:
                continue
                
            mag = magnitudes[idx]
            phase = np.angle(X_pos[idx])
            
            # Cosine and sine coefficients
            if freq == 0:  # DC
                cos_coeff = np.real(X_pos[idx]) / N
                sin_coeff = 0.0
                is_dc = True
            else:
                cos_coeff = 2 * np.real(X_pos[idx]) / N
                sin_coeff = -2 * np.imag(X_pos[idx]) / N
                is_dc = False
            
            self.frequency_set.append(freq)
            self.components.append(SpectralComponent(
                frequency=freq,
                magnitude=mag,
                phase=phase,
                cosine_coeff=cos_coeff,
                sine_coeff=sin_coeff,
                basis_index_cos=-1,  # Assigned in step 3
                basis_index_sin=-1 if is_dc else -1,
                is_dc=is_dc
            ))
        
        # Sort by frequency
        sorted_indices = np.argsort(self.frequency_set)
        self.frequency_set = [self.frequency_set[i] for i in sorted_indices]
        self.components = [self.components[i] for i in sorted_indices]
        
        # Step 3: Basis Assignment
        n_components = len(self.components)
        self.basis = GABasis(n_components, use_clifford=False)  # Use numpy for simplicity
        
        basis_idx = 0
        for comp in self.components:
            if comp.is_dc:
                comp.basis_index_cos = basis_idx
                comp.basis_index_sin = None
                basis_idx += 1
            else:
                comp.basis_index_cos = basis_idx
                comp.basis_index_sin = basis_idx + 1
                basis_idx += 2
        
        # Step 4: Vector Construction
        ga_vector = np.zeros(self.basis.dimension)
        
        for comp in self.components:
            if comp.is_dc:
                ga_vector[comp.basis_index_cos] = comp.cosine_coeff
            else:
                ga_vector[comp.basis_index_cos] = comp.cosine_coeff
                ga_vector[comp.basis_index_sin] = comp.sine_coeff
        
        return ga_vector, self.components
    
    def get_frequency_mapping(self) -> Dict[float, Tuple[int, Optional[int]]]:
        """
        Get mapping from frequency to basis indices.
        
        Returns
        -------
        mapping : dict
            Dictionary mapping frequency to (cos_index, sin_index) tuple.
        """
        mapping = {}
        for comp in self.components:
            mapping[comp.frequency] = (comp.basis_index_cos, comp.basis_index_sin)
        return mapping
    
    def inverse_transform(self, ga_vector: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Reconstruct time-domain signal from GA vector.
        
        Parameters
        ----------
        ga_vector : np.ndarray
            Geometric algebra vector.
        t : np.ndarray
            Time points for reconstruction.
            
        Returns
        -------
        signal : np.ndarray
            Reconstructed time-domain signal.
        """
        signal = np.zeros_like(t)
        
        for comp in self.components:
            if comp.is_dc:
                signal += ga_vector[comp.basis_index_cos]
            else:
                omega = 2 * np.pi * comp.frequency
                signal += ga_vector[comp.basis_index_cos] * np.cos(omega * t)
                signal += ga_vector[comp.basis_index_sin] * np.sin(omega * t)
        
        return signal
    
    def verify_parseval(self, signal: np.ndarray, ga_vector: np.ndarray) -> Dict[str, float]:
        """
        Verify Parseval's theorem: ||u||² = U_rms².
        
        Parameters
        ----------
        signal : np.ndarray
            Original time-domain signal.
        ga_vector : np.ndarray
            GA vector representation.
            
        Returns
        -------
        results : dict
            Dictionary with 'signal_rms_sq', 'vector_norm_sq', 'relative_error'.
        """
        # RMS from time domain
        signal_rms_sq = np.mean(signal**2)
        
        # Norm from GA vector (accounting for sqrt(2) factor in coefficients)
        # For sinusoidal: U_rms = U_peak/sqrt(2), coefficient = U_peak
        # So ||u||² = sum(coeff²/2) for AC + DC²
        vector_norm_sq = 0.0
        for comp in self.components:
            if comp.is_dc:
                vector_norm_sq += ga_vector[comp.basis_index_cos]**2
            else:
                vector_norm_sq += 0.5 * (ga_vector[comp.basis_index_cos]**2 + 
                                         ga_vector[comp.basis_index_sin]**2)
        
        relative_error = abs(signal_rms_sq - vector_norm_sq) / signal_rms_sq
        
        return {
            'signal_rms_sq': signal_rms_sq,
            'vector_norm_sq': vector_norm_sq,
            'relative_error': relative_error
        }


def index_mapping_function(frequency: float, frequency_set: List[float]) -> Tuple[int, Optional[int]]:
    """
    Index mapping function φ: F → N as defined in the paper.
    
    Parameters
    ----------
    frequency : float
        Frequency to map.
    frequency_set : list
        Ordered set of all frequencies.
        
    Returns
    -------
    indices : tuple
        (cosine_index, sine_index) where sine_index is None for DC.
        
    Notes
    -----
    Implements equation (eq:index_mapping) from the paper:
    φ(f_m) = 0 if f_m = 0 (DC)
           = 2m - 1 for cosine component at f_m > 0
           = 2m for sine component at f_m > 0
    """
    if frequency not in frequency_set:
        raise ValueError(f"Frequency {frequency} not in frequency set")
    
    m = frequency_set.index(frequency)
    
    if frequency == 0:  # DC
        return (0, None)
    else:
        # Account for DC at position 0
        has_dc = 0 in frequency_set
        offset = 1 if has_dc else 0
        
        # Position in AC frequencies
        ac_position = m - offset if has_dc else m
        
        cos_idx = offset + 2 * ac_position
        sin_idx = offset + 2 * ac_position + 1
        
        return (cos_idx, sin_idx)
