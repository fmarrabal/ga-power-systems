"""
Unit tests for basis.py - Fourier-to-GA mapping
"""

import numpy as np
import pytest
from gapot.basis import FourierToGA, GABasis, index_mapping_function


class TestGABasis:
    """Tests for GABasis class."""
    
    def test_dimension_calculation(self):
        """Test dimension calculation for different component numbers."""
        # 1 component (DC only)
        basis = GABasis(n_components=1, use_clifford=False)
        assert basis.dimension == 1
        
        # 2 components (DC + 1 AC)
        basis = GABasis(n_components=2, use_clifford=False)
        assert basis.dimension == 3  # 1 + 2*1
        
        # 5 components (DC + 4 AC)
        basis = GABasis(n_components=5, use_clifford=False)
        assert basis.dimension == 9  # 1 + 2*4
    
    def test_basis_orthonormality(self):
        """Test that basis vectors are orthonormal."""
        basis = GABasis(n_components=5, use_clifford=False)
        
        for i in range(basis.dimension):
            for j in range(basis.dimension):
                inner = basis.inner_product(
                    basis.basis_vectors[i], 
                    basis.basis_vectors[j]
                )
                expected = 1.0 if i == j else 0.0
                assert abs(inner - expected) < 1e-10
    
    def test_norm_calculation(self):
        """Test norm calculation."""
        basis = GABasis(n_components=3, use_clifford=False)
        v = np.array([3.0, 4.0, 0.0, 0.0, 0.0])
        assert abs(basis.norm(v) - 5.0) < 1e-10


class TestFourierToGA:
    """Tests for FourierToGA class."""
    
    def test_pure_sinusoid(self):
        """Test transformation of pure sinusoidal signal."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        # Pure 50 Hz sinusoid
        amplitude = 230 * np.sqrt(2)
        signal = amplitude * np.sin(2 * np.pi * f1 * t)
        
        converter = FourierToGA(fs=fs, f1=f1)
        ga_vec, components = converter.transform(signal)
        
        # Should have only one AC component at 50 Hz
        ac_components = [c for c in components if not c.is_dc]
        assert len(ac_components) >= 1
        
        # Check fundamental frequency
        fund_comp = next((c for c in components if abs(c.frequency - f1) < 1), None)
        assert fund_comp is not None
    
    def test_dc_component(self):
        """Test DC component detection."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        # DC + AC signal
        dc_value = 10.0
        signal = dc_value + 100 * np.sin(2 * np.pi * f1 * t)
        
        converter = FourierToGA(fs=fs, f1=f1)
        ga_vec, components = converter.transform(signal)
        
        # Should detect DC component
        dc_comp = next((c for c in components if c.is_dc), None)
        assert dc_comp is not None
        assert abs(dc_comp.cosine_coeff - dc_value) < 1.0
    
    def test_harmonic_detection(self):
        """Test detection of harmonic components."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        # Fundamental + 5th harmonic
        signal = (100 * np.sin(2 * np.pi * f1 * t) + 
                  20 * np.sin(2 * np.pi * 5 * f1 * t))
        
        converter = FourierToGA(fs=fs, f1=f1)
        ga_vec, components = converter.transform(signal)
        
        frequencies = [c.frequency for c in components]
        
        # Should detect both frequencies
        assert any(abs(f - f1) < 1 for f in frequencies)  # Fundamental
        assert any(abs(f - 5*f1) < 1 for f in frequencies)  # 5th harmonic
    
    def test_parseval_verification(self):
        """Test Parseval's theorem verification."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        signal = 230 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t)
        
        converter = FourierToGA(fs=fs, f1=f1)
        ga_vec, _ = converter.transform(signal)
        
        result = converter.verify_parseval(signal, ga_vec)
        
        # Relative error should be small
        assert result['relative_error'] < 0.01
    
    def test_inverse_transform(self):
        """Test inverse transformation reconstructs signal."""
        fs = 10000
        f1 = 50
        t = np.linspace(0, 0.04, int(0.04 * fs), endpoint=False)
        
        signal = (230 * np.sqrt(2) * np.sin(2 * np.pi * f1 * t) +
                  23 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f1 * t))
        
        converter = FourierToGA(fs=fs, f1=f1)
        ga_vec, _ = converter.transform(signal)
        
        reconstructed = converter.inverse_transform(ga_vec, t)
        
        # Reconstruction error should be small
        error = np.sqrt(np.mean((signal - reconstructed)**2))
        signal_rms = np.sqrt(np.mean(signal**2))
        
        assert error / signal_rms < 0.05


class TestIndexMapping:
    """Tests for index mapping function."""
    
    def test_dc_mapping(self):
        """Test DC component maps to index 0."""
        freq_set = [0, 50, 100, 150]
        idx_cos, idx_sin = index_mapping_function(0, freq_set)
        
        assert idx_cos == 0
        assert idx_sin is None
    
    def test_fundamental_mapping(self):
        """Test fundamental maps to indices 1, 2."""
        freq_set = [0, 50, 100, 150]
        idx_cos, idx_sin = index_mapping_function(50, freq_set)
        
        assert idx_cos == 1
        assert idx_sin == 2
    
    def test_harmonic_mapping(self):
        """Test harmonics map correctly."""
        freq_set = [0, 50, 100, 150]
        
        # 2nd harmonic
        idx_cos, idx_sin = index_mapping_function(100, freq_set)
        assert idx_cos == 3
        assert idx_sin == 4
        
        # 3rd harmonic
        idx_cos, idx_sin = index_mapping_function(150, freq_set)
        assert idx_cos == 5
        assert idx_sin == 6
    
    def test_invalid_frequency(self):
        """Test invalid frequency raises error."""
        freq_set = [0, 50, 100]
        
        with pytest.raises(ValueError):
            index_mapping_function(75, freq_set)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
