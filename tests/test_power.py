"""
Unit tests for power.py - Geometric power calculations
"""

import numpy as np
import pytest
from gapot.power import GeometricPower, calculate_geometric_power


class TestGeometricPower:
    """Tests for GeometricPower class."""
    
    @pytest.fixture
    def sinusoidal_signals(self):
        """Generate pure sinusoidal test signals."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        U = 230  # RMS voltage
        I = 10   # RMS current
        phi = np.pi / 6  # 30° lagging
        
        u = U * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
        i = I * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - phi)
        
        return u, i, f1, fs, U, I, phi
    
    @pytest.fixture
    def distorted_signals(self):
        """Generate distorted test signals with harmonics."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        # Fundamental
        U1, I1 = 230, 10
        phi1 = np.pi / 6
        
        u = U1 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
        i = I1 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - phi1)
        
        # Add 5th harmonic
        U5, I5 = 0.05 * U1, 0.20 * I1
        phi5 = 0.3
        u += U5 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f1 * t)
        i += I5 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f1 * t - phi5)
        
        return u, i, f1, fs
    
    def test_sinusoidal_active_power(self, sinusoidal_signals):
        """Test active power for pure sinusoidal case."""
        u, i, f1, fs, U, I, phi = sinusoidal_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        # Expected: P = U * I * cos(phi)
        P_expected = U * I * np.cos(phi)
        
        assert abs(gp.P - P_expected) / P_expected < 0.02
    
    def test_sinusoidal_reactive_power(self, sinusoidal_signals):
        """Test reactive power for pure sinusoidal case."""
        u, i, f1, fs, U, I, phi = sinusoidal_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        # Expected: Q = U * I * sin(phi)
        Q_expected = U * I * np.sin(phi)
        
        assert abs(gp.M_Q_norm - Q_expected) / Q_expected < 0.02
    
    def test_sinusoidal_power_factor(self, sinusoidal_signals):
        """Test power factor for pure sinusoidal case."""
        u, i, f1, fs, U, I, phi = sinusoidal_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        # Expected: PF = cos(phi)
        PF_expected = np.cos(phi)
        
        assert abs(gp.PF - PF_expected) < 0.02
    
    def test_sinusoidal_no_distortion(self, sinusoidal_signals):
        """Test that pure sinusoidal has no distortion power."""
        u, i, f1, fs, _, _, _ = sinusoidal_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        # Distortion should be negligible
        assert gp.M_D_norm < 1.0  # Less than 1 VA
    
    def test_distorted_has_distortion_power(self, distorted_signals):
        """Test that distorted signals have distortion power."""
        u, i, f1, fs = distorted_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        # Should have measurable distortion
        assert gp.M_D_norm > 0.1
    
    def test_energy_conservation(self, sinusoidal_signals):
        """Test energy conservation: ||M||² = P² + ||M_Q||² + ||M_D||²."""
        u, i, f1, fs, _, _, _ = sinusoidal_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        # ||M||² should equal sum of squared components
        S_sq = gp.S ** 2
        components_sq = gp.P**2 + gp.M_Q_norm**2 + gp.M_D_norm**2
        
        assert abs(S_sq - components_sq) / S_sq < 0.01
    
    def test_apparent_power_consistency(self, sinusoidal_signals):
        """Test S_geometric ≈ S_traditional = U_rms * I_rms."""
        u, i, f1, fs, _, _, _ = sinusoidal_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        verification = gp.verify_energy_conservation()
        
        assert verification['relative_error'] < 0.02
    
    def test_harmonic_power_breakdown(self, distorted_signals):
        """Test harmonic power breakdown."""
        u, i, f1, fs = distorted_signals
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        harmonic_powers = gp.get_harmonic_powers()
        
        # Should have entries for fundamental and 5th harmonic
        assert any(abs(f - f1) < 1 for f in harmonic_powers.keys())
        assert any(abs(f - 5*f1) < 5 for f in harmonic_powers.keys())
        
        # Total P should approximately equal sum of harmonic P's
        P_sum = sum(pwr['P'] for pwr in harmonic_powers.values())
        assert abs(gp.P - P_sum) / abs(gp.P) < 0.05
    
    def test_resistive_load(self):
        """Test purely resistive load (PF = 1)."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        u = 230 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
        i = 10 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)  # In phase
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        assert abs(gp.PF - 1.0) < 0.02
        assert gp.M_Q_norm < 10  # Small reactive power
    
    def test_reactive_load(self):
        """Test purely reactive load (PF ≈ 0)."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        u = 230 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
        i = 10 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - np.pi/2)  # 90° lagging
        
        gp = GeometricPower(u, i, f1=f1, fs=fs)
        
        assert abs(gp.PF) < 0.05  # Nearly zero
        assert abs(gp.P) < 50  # Small active power


class TestConvenienceFunction:
    """Tests for calculate_geometric_power function."""
    
    def test_basic_calculation(self):
        """Test basic power calculation."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        u = 230 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
        i = 10 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t - np.pi/6)
        
        result = calculate_geometric_power(u, i, f1=f1, fs=fs)
        
        assert 'P' in result
        assert 'Q' in result
        assert 'D' in result
        assert 'S' in result
        assert 'PF' in result
        
        # Check reasonable values
        assert 1500 < result['P'] < 2500
        assert 0 < result['PF'] < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
