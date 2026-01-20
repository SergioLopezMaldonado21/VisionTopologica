"""Tests para PatchExtractor."""

import numpy as np
import pytest

from patches_tda.extraction.patch_extractor import PatchExtractor


class TestPatchExtractor:
    """Tests para la clase PatchExtractor."""
    
    def test_extract_shape(self):
        """Debe retornar parches con shape correcto."""
        extractor = PatchExtractor(patch_size=3, n_patches=100)
        image = np.random.rand(100, 100)
        
        patches = extractor.extract(image)
        
        assert patches.shape == (100, 9)
        assert patches.dtype == np.float64
    
    def test_extract_reproducibility(self):
        """Debe ser reproducible con la misma semilla."""
        image = np.random.rand(100, 100)
        
        ext1 = PatchExtractor(patch_size=3, n_patches=50, rng=np.random.default_rng(42))
        ext2 = PatchExtractor(patch_size=3, n_patches=50, rng=np.random.default_rng(42))
        
        p1 = ext1.extract(image)
        p2 = ext2.extract(image)
        
        np.testing.assert_array_equal(p1, p2)
    
    def test_patches_are_centered(self):
        """Los parches deben estar centrados (media ≈ 0)."""
        extractor = PatchExtractor(patch_size=3, n_patches=100, rng=np.random.default_rng(42))
        image = np.random.rand(100, 100) * 1000  # Valores grandes
        
        patches = extractor.extract(image)
        means = patches.mean(axis=1)
        
        # Cada parche debe tener media cercana a 0
        np.testing.assert_array_almost_equal(means, np.zeros(100), decimal=10)
    
    def test_log_intensity_applied(self):
        """Debe aplicar log(1+x) antes de centrar."""
        extractor = PatchExtractor(patch_size=2, n_patches=1, rng=np.random.default_rng(0))
        # Imagen con valores conocidos
        image = np.array([[0, 1], [2, 3]], dtype=np.float64)
        
        patches = extractor.extract(image)
        
        # log(1 + [0,1,2,3]) = [0, log(2), log(3), log(4)]
        expected_before_center = np.array([0, np.log(2), np.log(3), np.log(4)])
        expected = expected_before_center - expected_before_center.mean()
        
        np.testing.assert_array_almost_equal(patches[0], expected)
    
    def test_invalid_image_raises(self):
        """Debe lanzar error con imagen 1D."""
        extractor = PatchExtractor(patch_size=3, n_patches=100)
        
        with pytest.raises(ValueError, match="debe ser 2D"):
            extractor.extract(np.array([1, 2, 3]))
    
    def test_image_too_small_raises(self):
        """Debe lanzar error si imagen es muy pequeña."""
        extractor = PatchExtractor(patch_size=5, n_patches=100)
        
        with pytest.raises(ValueError, match="muy pequeña"):
            extractor.extract(np.random.rand(3, 3))
