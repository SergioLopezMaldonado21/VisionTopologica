"""Tests para DensityFilter."""

import numpy as np
import pytest

from patches_tda.filtering.density_filter import DensityFilter


class TestDensityFilter:
    """Tests para la clase DensityFilter."""
    
    def test_transform_reduces_points(self):
        """transform() debe reducir el número de puntos."""
        filt = DensityFilter(
            k_neighbors=5,
            top_density_fraction=0.30,
            denoise_iterations=1
        )
        X = np.random.randn(1000, 9)
        
        X_pk = filt.transform(X)
        
        assert X_pk.shape[0] < X.shape[0]
        assert X_pk.shape[0] == pytest.approx(300, abs=10)  # ~30% de 1000
        assert X_pk.shape[1] == 9
    
    def test_estimate_density_shape(self):
        """estimate_density() debe retornar array del tamaño correcto."""
        filt = DensityFilter(k_neighbors=10)
        X = np.random.randn(500, 9)
        
        densities = filt.estimate_density(X)
        
        assert densities.shape == (500,)
        assert np.all(densities > 0)
    
    def test_filter_by_density_selects_correct_fraction(self):
        """filter_by_density() debe seleccionar la fracción correcta."""
        filt = DensityFilter(k_neighbors=5, top_density_fraction=0.25)
        X = np.random.randn(400, 9)
        
        X_filtered = filt.filter_by_density(X)
        
        assert X_filtered.shape[0] == 100  # 25% de 400
    
    def test_denoise_with_zero_iterations_returns_copy(self):
        """denoise() con 0 iteraciones debe retornar copia."""
        filt = DensityFilter(denoise_iterations=0)
        X = np.random.randn(100, 9)
        
        X_denoised = filt.denoise(X)
        
        np.testing.assert_array_equal(X, X_denoised)
        assert X is not X_denoised  # Debe ser una copia
    
    def test_denoise_smooths_data(self):
        """denoise() debe suavizar los datos."""
        filt = DensityFilter(k_neighbors=5, denoise_iterations=2)
        
        # Datos con ruido
        X = np.random.randn(100, 3)
        
        X_denoised = filt.denoise(X)
        
        # La varianza debería reducirse con denoising
        assert X_denoised.var() < X.var() * 0.9
    
    def test_invalid_parameters_raise(self):
        """Debe lanzar error con parámetros inválidos."""
        with pytest.raises(ValueError):
            DensityFilter(k_neighbors=0)
        
        with pytest.raises(ValueError):
            DensityFilter(top_density_fraction=1.5)
        
        with pytest.raises(ValueError):
            DensityFilter(denoise_iterations=-1)
