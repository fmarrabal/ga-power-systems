"""
Pytest configuration and fixtures for GAPoT tests.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def sample_signals():
    """Generate standard test signals."""
    fs = 10000
    f1 = 50
    t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
    
    U, I = 230, 10
    phi = np.pi / 6
    
    u = U * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
    i = I * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - phi)
    
    return {
        'u': u,
        'i': i,
        't': t,
        'f1': f1,
        'fs': fs,
        'U': U,
        'I': I,
        'phi': phi
    }


@pytest.fixture
def distorted_signals():
    """Generate distorted test signals with harmonics."""
    fs = 10000
    f1 = 50
    t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
    
    U1, I1 = 230, 10
    phi1 = np.pi / 6
    
    u = U1 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
    i = I1 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - phi1)
    
    # 5th harmonic
    u += 0.05 * U1 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f1 * t)
    i += 0.20 * I1 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f1 * t - 0.3)
    
    return {
        'u': u,
        'i': i,
        't': t,
        'f1': f1,
        'fs': fs
    }


@pytest.fixture
def three_phase_signals():
    """Generate balanced three-phase signals."""
    fs = 10000
    f1 = 50
    t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
    
    U = 230
    omega = 2 * np.pi * f1
    
    v_a = U * np.sqrt(2) * np.sin(omega * t)
    v_b = U * np.sqrt(2) * np.sin(omega * t - 2*np.pi/3)
    v_c = U * np.sqrt(2) * np.sin(omega * t + 2*np.pi/3)
    
    return {
        'v_a': v_a,
        'v_b': v_b,
        'v_c': v_c,
        't': t,
        'f1': f1,
        'fs': fs,
        'U': U
    }
