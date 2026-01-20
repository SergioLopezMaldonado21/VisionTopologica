"""
IMCImageLoader - Carga de imágenes Van Hateren en formato .imc

Formato .imc:
- Binario crudo, sin encabezado
- Tipo de dato: uint16 (big-endian)
- Resolución: 1024 × 1536 píxeles
- Escala de grises
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# Constantes del formato .imc
IMC_HEIGHT = 1024
IMC_WIDTH = 1536
IMC_EXPECTED_BYTES = IMC_HEIGHT * IMC_WIDTH * 2  # uint16 = 2 bytes


class IMCImageLoader:
    """
    Cargador de imágenes Van Hateren en formato .imc.
    
    El formato .imc es binario crudo con las siguientes características:
    - Tipo de dato: uint16 (big-endian)
    - Resolución: 1024 × 1536 píxeles
    - Sin encabezado ni metadata
    
    Parameters
    ----------
    data_dir : Path | str
        Directorio que contiene los archivos .imc.
    
    Raises
    ------
    FileNotFoundError
        Si el directorio no existe.
    
    Examples
    --------
    >>> loader = IMCImageLoader(Path("vanhateren_imc"))
    >>> image = loader.load("imk00001.imc")
    >>> image.shape
    (1024, 1536)
    >>> image.dtype
    dtype('float64')
    """
    
    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self._data_dir}")
        if not self._data_dir.is_dir():
            raise NotADirectoryError(f"No es un directorio: {self._data_dir}")
        
        logger.debug("IMCImageLoader inicializado con data_dir=%s", self._data_dir)
    
    @property
    def data_dir(self) -> Path:
        """Directorio de datos."""
        return self._data_dir
    
    def load(self, filename: str) -> np.ndarray:
        """
        Carga una imagen .imc y la retorna como array float64.
        
        Parameters
        ----------
        filename : str
            Nombre del archivo .imc (e.g., "imk00001.imc").
        
        Returns
        -------
        np.ndarray
            Imagen como array de shape (1024, 1536) y dtype float64.
        
        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        ValueError
            Si el tamaño del archivo no es correcto.
        """
        filepath = self._data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        # Validar tamaño del archivo
        file_size = filepath.stat().st_size
        if file_size != IMC_EXPECTED_BYTES:
            raise ValueError(
                f"Tamaño de archivo incorrecto: {file_size} bytes "
                f"(esperado: {IMC_EXPECTED_BYTES} bytes)"
            )
        
        # Leer como uint16 big-endian
        raw_data = np.fromfile(filepath, dtype=">u2")  # big-endian uint16
        
        # Reshape a imagen 2D
        image = raw_data.reshape((IMC_HEIGHT, IMC_WIDTH))
        
        # Convertir a float64 para procesamiento
        image = image.astype(np.float64)
        
        logger.debug("Imagen cargada: %s, shape=%s", filename, image.shape)
        
        return image
    
    def list_images(self) -> list[str]:
        """
        Lista todos los archivos .imc en el directorio.
        
        Returns
        -------
        list[str]
            Lista ordenada de nombres de archivo .imc.
        """
        files = sorted(f.name for f in self._data_dir.glob("*.imc"))
        return files
    
    def iter_images(
        self, 
        limit: int | None = None
    ) -> Iterator[tuple[str, np.ndarray]]:
        """
        Itera sobre las imágenes del directorio.
        
        Parameters
        ----------
        limit : int | None, optional
            Número máximo de imágenes a cargar. None para todas.
        
        Yields
        ------
        tuple[str, np.ndarray]
            Tupla (nombre_archivo, imagen).
        """
        filenames = self.list_images()
        
        if limit is not None:
            filenames = filenames[:limit]
        
        for filename in filenames:
            try:
                image = self.load(filename)
                yield filename, image
            except (ValueError, OSError) as e:
                logger.warning("Error cargando %s: %s", filename, e)
                continue
