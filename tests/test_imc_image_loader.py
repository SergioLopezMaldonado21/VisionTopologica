"""Tests para IMCImageLoader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from patches_tda.io.imc_image_loader import IMCImageLoader, IMC_HEIGHT, IMC_WIDTH


class TestIMCImageLoader:
    """Tests para la clase IMCImageLoader."""
    
    def test_init_with_invalid_dir_raises(self):
        """Debe lanzar error si el directorio no existe."""
        with pytest.raises(FileNotFoundError):
            IMCImageLoader(Path("/nonexistent/path"))
    
    def test_load_valid_imc(self, tmp_path: Path):
        """Debe cargar correctamente un archivo .imc válido."""
        # Crear archivo de prueba
        test_data = np.random.randint(0, 65535, (IMC_HEIGHT, IMC_WIDTH), dtype=np.uint16)
        filepath = tmp_path / "test.imc"
        # Guardar en big-endian
        test_data.astype(">u2").tofile(filepath)
        
        loader = IMCImageLoader(tmp_path)
        image = loader.load("test.imc")
        
        assert image.shape == (IMC_HEIGHT, IMC_WIDTH)
        assert image.dtype == np.float64
        np.testing.assert_array_equal(image.astype(np.uint16), test_data)
    
    def test_load_invalid_size_raises(self, tmp_path: Path):
        """Debe lanzar error si el archivo tiene tamaño incorrecto."""
        filepath = tmp_path / "bad.imc"
        # Archivo con tamaño incorrecto
        np.array([1, 2, 3], dtype=np.uint16).tofile(filepath)
        
        loader = IMCImageLoader(tmp_path)
        with pytest.raises(ValueError, match="Tamaño de archivo incorrecto"):
            loader.load("bad.imc")
    
    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        """Debe lanzar error si el archivo no existe."""
        loader = IMCImageLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.imc")
    
    def test_list_images(self, tmp_path: Path):
        """Debe listar correctamente los archivos .imc."""
        # Crear archivos de prueba
        (tmp_path / "img1.imc").touch()
        (tmp_path / "img2.imc").touch()
        (tmp_path / "other.txt").touch()
        
        loader = IMCImageLoader(tmp_path)
        files = loader.list_images()
        
        assert files == ["img1.imc", "img2.imc"]
