"""Tests para DNormCalculator."""

import numpy as np
import pytest

from patches_tda.dnorm.dnorm_calculator import DNormCalculator


class TestDNormCalculator:
    """Tests para la clase DNormCalculator."""
    
    def test_d_matrix_shape(self):
        """La matriz D debe tener shape (m², m²)."""
        calc = DNormCalculator(patch_size=3)
        assert calc.D_matrix.shape == (9, 9)
        
        calc = DNormCalculator(patch_size=5)
        assert calc.D_matrix.shape == (25, 25)
    
    def test_d_matrix_is_symmetric(self):
        """La matriz D debe ser simétrica."""
        calc = DNormCalculator(patch_size=3)
        D = calc.D_matrix
        
        np.testing.assert_array_almost_equal(D, D.T)
    
    def test_d_matrix_is_positive_semidefinite(self):
        """La matriz D debe ser semidefinida positiva."""
        calc = DNormCalculator(patch_size=3)
        D = calc.D_matrix
        
        eigenvalues = np.linalg.eigvalsh(D)
        assert np.all(eigenvalues >= -1e-10)  # Tolerancia numérica
    
    def test_compute_returns_nonnegative(self):
        """D-norm debe ser no negativa."""
        calc = DNormCalculator(patch_size=3)
        patches = np.random.randn(100, 9)
        
        dnorms = calc.compute(patches)
        
        assert np.all(dnorms >= 0)
    
    def test_uniform_patch_has_zero_dnorm(self):
        """Un parche uniforme debe tener D-norm = 0."""
        calc = DNormCalculator(patch_size=3)
        uniform_patch = np.ones((1, 9)) * 5.0
        
        dnorm = calc.compute(uniform_patch)
        
        assert dnorm[0] == pytest.approx(0.0, abs=1e-10)
    
    def test_high_contrast_has_high_dnorm(self):
        """Un parche con alto contraste debe tener D-norm alta."""
        calc = DNormCalculator(patch_size=3)
        
        # Parche uniforme (bajo contraste)
        low_contrast = np.ones((1, 9)) * 5.0
        
        # Parche con bordes fuertes (alto contraste)
        high_contrast = np.array([[1, 1, 1, 5, 5, 5, 9, 9, 9]])
        
        dnorm_low = calc.compute(low_contrast)
        dnorm_high = calc.compute(high_contrast)
        
        assert dnorm_high[0] > dnorm_low[0]
    
    def test_compute_wrong_dim_raises(self):
        """Debe lanzar error si dimensión no coincide."""
        calc = DNormCalculator(patch_size=3)
        
        with pytest.raises(ValueError, match="Dimensión de parche incorrecta"):
            calc.compute(np.random.randn(10, 16))  # 16 != 9
