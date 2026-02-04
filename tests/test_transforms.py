"""
Unit tests for transforms.py - GA-based electrical transformations
"""

import numpy as np
import pytest
from gapot.transforms import (
    Rotor, ClarkeTransform, ParkTransform, FortescueTransform,
    rotation_matrix_from_rotor, rotor_from_rotation_matrix, slerp
)


class TestRotor:
    """Tests for Rotor class."""
    
    def test_identity_rotation(self):
        """Test zero angle gives identity transformation."""
        rotor = Rotor(angle=0, plane=(0, 1))
        v = np.array([1.0, 2.0, 3.0])
        v_rotated = rotor.apply(v)
        
        np.testing.assert_array_almost_equal(v, v_rotated)
    
    def test_90_degree_rotation(self):
        """Test 90° rotation in xy-plane."""
        rotor = Rotor(angle=np.pi/2, plane=(0, 1))
        v = np.array([1.0, 0.0, 0.0])
        v_rotated = rotor.apply(v)
        
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(v_rotated, expected, decimal=10)
    
    def test_180_degree_rotation(self):
        """Test 180° rotation."""
        rotor = Rotor(angle=np.pi, plane=(0, 1))
        v = np.array([1.0, 0.0, 0.0])
        v_rotated = rotor.apply(v)
        
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(v_rotated, expected, decimal=10)
    
    def test_rotor_reverse(self):
        """Test rotor reverse undoes rotation."""
        rotor = Rotor(angle=np.pi/4, plane=(0, 1))
        v = np.array([1.0, 2.0, 3.0])
        
        v_rotated = rotor.apply(v)
        v_back = rotor.reverse.apply(v_rotated)
        
        np.testing.assert_array_almost_equal(v, v_back)
    
    def test_rotor_composition(self):
        """Test rotor composition."""
        r1 = Rotor(angle=np.pi/6, plane=(0, 1))
        r2 = Rotor(angle=np.pi/4, plane=(0, 1))
        
        r_combined = r1.compose(r2)
        
        assert abs(r_combined.angle - (np.pi/6 + np.pi/4)) < 1e-10
    
    def test_norm_preservation(self):
        """Test that rotation preserves vector norm."""
        rotor = Rotor(angle=0.7, plane=(0, 1))
        v = np.array([3.0, 4.0, 5.0])
        v_rotated = rotor.apply(v)
        
        assert abs(np.linalg.norm(v) - np.linalg.norm(v_rotated)) < 1e-10


class TestClarkeTransform:
    """Tests for Clarke (αβ) transformation."""
    
    def test_balanced_three_phase(self):
        """Test Clarke transform on balanced three-phase."""
        clarke = ClarkeTransform(amplitude_invariant=False)
        
        t = np.linspace(0, 0.02, 1000)
        omega = 2 * np.pi * 50
        
        v_a = 230 * np.sqrt(2) * np.sin(omega * t)
        v_b = 230 * np.sqrt(2) * np.sin(omega * t - 2*np.pi/3)
        v_c = 230 * np.sqrt(2) * np.sin(omega * t + 2*np.pi/3)
        
        v_alpha, v_beta = clarke.forward(v_a, v_b, v_c)
        
        # For balanced system, alpha and beta should be 90° apart
        # Check they have similar magnitudes
        alpha_rms = np.sqrt(np.mean(v_alpha**2))
        beta_rms = np.sqrt(np.mean(v_beta**2))
        
        assert abs(alpha_rms - beta_rms) / alpha_rms < 0.05
    
    def test_inverse_recovery(self):
        """Test inverse transform recovers original."""
        clarke = ClarkeTransform(amplitude_invariant=True)
        
        t = np.linspace(0, 0.02, 1000)
        omega = 2 * np.pi * 50
        
        v_a = 230 * np.sqrt(2) * np.sin(omega * t)
        v_b = 230 * np.sqrt(2) * np.sin(omega * t - 2*np.pi/3)
        v_c = 230 * np.sqrt(2) * np.sin(omega * t + 2*np.pi/3)
        
        v_alpha, v_beta = clarke.forward(v_a, v_b, v_c)
        v_a_rec, v_b_rec, v_c_rec = clarke.inverse(v_alpha, v_beta)
        
        # Note: Recovery loses zero-sequence, but for balanced it's okay
        np.testing.assert_array_almost_equal(v_a, v_a_rec, decimal=1)


class TestParkTransform:
    """Tests for Park (dq) transformation."""
    
    def test_stationary_dq(self):
        """Test Park transform gives DC values for rotating frame."""
        omega = 2 * np.pi * 50
        park = ParkTransform(omega=omega)
        
        t = np.linspace(0, 0.04, 2000)
        
        # Rotating vector in αβ
        v_alpha = 230 * np.sqrt(2) * np.cos(omega * t)
        v_beta = 230 * np.sqrt(2) * np.sin(omega * t)
        
        v_d, v_q = park.forward(v_alpha, v_beta, t)
        
        # In synchronous frame, should be approximately DC
        v_d_std = np.std(v_d)
        v_q_std = np.std(v_q)
        
        # Standard deviation should be small compared to mean
        assert v_d_std / np.mean(np.abs(v_d)) < 0.1
    
    def test_inverse_recovery(self):
        """Test inverse transform recovers original."""
        omega = 2 * np.pi * 50
        park = ParkTransform(omega=omega)
        
        t = np.linspace(0, 0.04, 2000)
        
        v_alpha = 230 * np.cos(omega * t + 0.5)
        v_beta = 230 * np.sin(omega * t + 0.5)
        
        v_d, v_q = park.forward(v_alpha, v_beta, t)
        v_alpha_rec, v_beta_rec = park.inverse(v_d, v_q, t)
        
        np.testing.assert_array_almost_equal(v_alpha, v_alpha_rec, decimal=5)
        np.testing.assert_array_almost_equal(v_beta, v_beta_rec, decimal=5)
    
    def test_rotor_angle(self):
        """Test rotor angle calculation."""
        omega = 2 * np.pi * 50
        theta0 = np.pi / 4
        park = ParkTransform(omega=omega, theta0=theta0)
        
        t = np.array([0.0, 0.005, 0.01])
        angles = park.get_rotor_angle(t)
        
        expected = omega * t + theta0
        np.testing.assert_array_almost_equal(angles, expected)


class TestFortescueTransform:
    """Tests for Fortescue (symmetrical components) transformation."""
    
    def test_balanced_positive_sequence(self):
        """Test balanced system has only positive sequence."""
        fortescue = FortescueTransform()
        
        # Balanced phasors (complex)
        V = 230
        v_a = V * np.exp(1j * 0)
        v_b = V * np.exp(1j * (-2*np.pi/3))
        v_c = V * np.exp(1j * (2*np.pi/3))
        
        v_pos, v_neg, v_zero = fortescue.forward(v_a, v_b, v_c)
        
        # Positive sequence should equal V
        assert abs(np.abs(v_pos) - V) / V < 0.01
        
        # Negative and zero should be negligible
        assert np.abs(v_neg) / V < 0.01
        assert np.abs(v_zero) / V < 0.01


class TestRotorMatrixConversion:
    """Tests for rotor-matrix conversion functions."""
    
    def test_rotor_to_matrix(self):
        """Test conversion from rotor to matrix."""
        angle = np.pi / 4
        rotor = Rotor(angle=angle, plane=(0, 1))
        
        R = rotation_matrix_from_rotor(rotor, dimension=3)
        
        # Check it's a rotation matrix
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))
    
    def test_matrix_to_rotor(self):
        """Test conversion from matrix to rotor."""
        angle = np.pi / 3
        rotor_orig = Rotor(angle=angle, plane=(0, 1))
        
        R = rotation_matrix_from_rotor(rotor_orig, dimension=3)
        rotor_rec = rotor_from_rotation_matrix(R, plane=(0, 1))
        
        assert abs(rotor_orig.angle - rotor_rec.angle) < 1e-10
    
    def test_roundtrip_consistency(self):
        """Test rotor → matrix → rotor roundtrip."""
        angle = 0.7
        rotor = Rotor(angle=angle, plane=(0, 1))
        v = np.array([1.0, 2.0, 3.0])
        
        # Apply via rotor
        v_rotor = rotor.apply(v)
        
        # Apply via matrix
        R = rotation_matrix_from_rotor(rotor, dimension=3)
        v_matrix = R @ v
        
        np.testing.assert_array_almost_equal(v_rotor, v_matrix)


class TestSlerp:
    """Tests for spherical linear interpolation."""
    
    def test_slerp_endpoints(self):
        """Test SLERP at endpoints."""
        r1 = Rotor(angle=0, plane=(0, 1))
        r2 = Rotor(angle=np.pi/2, plane=(0, 1))
        
        r_start = slerp(r1, r2, 0.0)
        r_end = slerp(r1, r2, 1.0)
        
        assert abs(r_start.angle - r1.angle) < 1e-10
        assert abs(r_end.angle - r2.angle) < 1e-10
    
    def test_slerp_midpoint(self):
        """Test SLERP at midpoint."""
        r1 = Rotor(angle=0, plane=(0, 1))
        r2 = Rotor(angle=np.pi/2, plane=(0, 1))
        
        r_mid = slerp(r1, r2, 0.5)
        
        expected_angle = np.pi / 4
        assert abs(r_mid.angle - expected_angle) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
