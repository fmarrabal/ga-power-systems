"""
GAPoT - Geometric Algebra Power Theory Framework
================================================

A Python implementation of Geometric Algebra methods for electrical
power systems analysis.

Main Classes
------------
GeometricPower : Calculate geometric power from voltage and current signals
FourierToGA : Transform time-domain signals to geometric algebra vectors
CurrentDecomposition : Decompose current into active and non-active components
IEEE1459Power : IEEE 1459 standard power calculations for comparison
PQTheory : Instantaneous power theory (p-q) calculations

Transforms
----------
ClarkeTransform : abc to αβ transformation using rotors
ParkTransform : αβ to dq transformation using rotors
FortescueTransform : Symmetrical components using GA

Noise Mitigation
----------------
HoloborodkoFilter : Smooth noise-robust differentiator
TikhonovRegularizer : L2 regularization for derivatives
TVRegularizer : Total Variation regularization

Example
-------
>>> import numpy as np
>>> from gapot import GeometricPower
>>> t = np.linspace(0, 0.04, 2000)
>>> u = 230*np.sqrt(2) * np.sin(2*np.pi*50*t)
>>> i = 10*np.sqrt(2) * np.sin(2*np.pi*50*t - 0.5)
>>> gp = GeometricPower(u, i, f1=50, fs=50000)
>>> print(f"P = {gp.P:.2f} W, Q = {gp.M_Q_norm:.2f} var")

References
----------
.. [1] Montoya et al. (2026). Geometric Algebra in Electrical Power 
       Engineering. Phil. Trans. R. Soc. A.
.. [2] Menti et al. (2007). Geometric algebra: A powerful tool for 
       representing power under nonsinusoidal conditions.
.. [3] Castro-Núñez & Castro-Puche (2012). The IEEE Standard 1459, 
       the CPC Power Theory, and Geometric Algebra.

"""

__version__ = "1.0.0"
__author__ = "Francisco G. Montoya, Alfredo Alcayde, Francisco M. Arrabal-Campos"
__email__ = "pagilm@ual.es"

# Core classes
from .power import GeometricPower
from .basis import FourierToGA, GABasis
from .compensation import CurrentDecomposition
from .traditional import IEEE1459Power, PQTheory

# Transforms
from .transforms import ClarkeTransform, ParkTransform, FortescueTransform

# Noise mitigation
from .noise import HoloborodkoFilter, TikhonovRegularizer, TVRegularizer

# Utilities
from .utils import generate_distorted_signal, rms, thd

__all__ = [
    # Core
    "GeometricPower",
    "FourierToGA",
    "GABasis",
    "CurrentDecomposition",
    # Traditional methods
    "IEEE1459Power",
    "PQTheory",
    # Transforms
    "ClarkeTransform",
    "ParkTransform",
    "FortescueTransform",
    # Noise
    "HoloborodkoFilter",
    "TikhonovRegularizer",
    "TVRegularizer",
    # Utilities
    "generate_distorted_signal",
    "rms",
    "thd",
]
