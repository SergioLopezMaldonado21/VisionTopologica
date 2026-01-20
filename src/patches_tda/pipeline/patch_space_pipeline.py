"""
PatchSpacePipeline - Orquestador del pipeline completo

Compone todas las etapas para construir X(p,k) a partir
de un directorio de imágenes .imc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from patches_tda.io.imc_image_loader import IMCImageLoader
from patches_tda.extraction.patch_extractor import PatchExtractor
from patches_tda.extraction.top_dnorm_selector import TopDNormSelector
from patches_tda.dnorm.dnorm_calculator import DNormCalculator
from patches_tda.space.patch_space_builder import PatchSpaceBuilder
from patches_tda.space.global_subsampler import GlobalSubsampler
from patches_tda.filtering.density_filter import DensityFilter

if TYPE_CHECKING:
    from numpy.random import Generator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuración del pipeline completo.
    
    Attributes
    ----------
    data_dir : Path
        Directorio con archivos .imc.
    patch_size : int
        Tamaño del parche (m×m).
    patches_per_image : int
        Parches a extraer por imagen (Y).
    top_dnorm_fraction : float
        Fracción top por D-norm (q).
    target_patch_space_size : int
        Tamaño objetivo del patch space (Z_target).
    subsample_size : int
        Tamaño del submuestreo para topología (N).
    density_k : int
        Vecinos para filtrado por densidad (k).
    density_top_fraction : float
        Fracción de puntos densos a conservar (p).
    denoise_iterations : int
        Iteraciones de denoising.
    seed : int
        Semilla para reproducibilidad.
    max_images : int | None
        Límite de imágenes a procesar (None = todas).
    """
    data_dir: Path
    patch_size: int = 3
    patches_per_image: int = 5000
    top_dnorm_fraction: float = 0.20
    target_patch_space_size: int = 4_000_000
    subsample_size: int = 50_000
    density_k: int = 15
    density_top_fraction: float = 0.30
    denoise_iterations: int = 2
    seed: int = 42
    max_images: int | None = None


@dataclass
class PipelineStats:
    """Estadísticas del pipeline."""
    images_processed: int = 0
    total_patches_extracted: int = 0
    patches_after_dnorm: int = 0
    patch_space_size: int = 0
    subsample_size: int = 0
    final_size: int = 0


class PatchSpacePipeline:
    """
    Orquesta el pipeline completo de construcción del patch space.
    
    Flujo:
    1. Iterar imágenes .imc
    2. Extraer parches (log-intensidad + centrado)
    3. Seleccionar top por D-norm (por imagen)
    4. Acumular en patch space 𝓜
    5. Submuestrear globalmente → X
    6. Filtrar por densidad + denoise → X(p,k)
    
    Parameters
    ----------
    config : PipelineConfig
        Configuración del pipeline.
    
    Examples
    --------
    >>> config = PipelineConfig(data_dir=Path("vanhateren_imc"))
    >>> pipeline = PatchSpacePipeline(config)
    >>> X_pk = pipeline.run()
    >>> print(pipeline.last_stats)
    """
    
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._rng: Generator = np.random.default_rng(config.seed)
        self._stats = PipelineStats()
        
        # Inicializar componentes
        self._loader = IMCImageLoader(config.data_dir)
        self._extractor = PatchExtractor(
            patch_size=config.patch_size,
            n_patches=config.patches_per_image,
            rng=self._rng
        )
        self._dnorm_calc = DNormCalculator(patch_size=config.patch_size)
        self._selector = TopDNormSelector(
            dnorm_calculator=self._dnorm_calc,
            top_fraction=config.top_dnorm_fraction
        )
        self._builder = PatchSpaceBuilder(
            target_size=config.target_patch_space_size,
            patch_dim=config.patch_size ** 2
        )
        self._subsampler = GlobalSubsampler(
            n_samples=config.subsample_size,
            rng=self._rng
        )
        self._density_filter = DensityFilter(
            k_neighbors=config.density_k,
            top_density_fraction=config.density_top_fraction,
            denoise_iterations=config.denoise_iterations
        )
        
        logger.info("PatchSpacePipeline inicializado con config: %s", config)
    
    @property
    def config(self) -> PipelineConfig:
        """Configuración del pipeline."""
        return self._config
    
    @property
    def last_stats(self) -> PipelineStats:
        """Estadísticas de la última ejecución."""
        return self._stats
    
    def run(self) -> np.ndarray:
        """
        Ejecuta el pipeline completo y retorna X(p,k).
        
        Returns
        -------
        np.ndarray
            Conjunto final X(p,k) de shape (M, patch_dim).
        """
        logger.info("Iniciando pipeline...")
        self._stats = PipelineStats()
        self._builder.reset()
        
        # Etapa 1-4: Procesar imágenes y construir patch space
        logger.info("Etapa 1-4: Construyendo patch space...")
        for filename, image in self._loader.iter_images(limit=self._config.max_images):
            if self._builder.is_full:
                logger.info("Patch space lleno, deteniendo procesamiento de imágenes")
                break
            
            # Extraer parches
            patches = self._extractor.extract(image)
            self._stats.total_patches_extracted += patches.shape[0]
            
            # Seleccionar top por D-norm
            selected = self._selector.select(patches)
            self._stats.patches_after_dnorm += selected.shape[0]
            
            # Agregar al patch space
            self._builder.add_patches(selected)
            self._stats.images_processed += 1
            
            if self._stats.images_processed % 100 == 0:
                logger.info(
                    "Procesadas %d imágenes, patch space: %d/%d",
                    self._stats.images_processed,
                    self._builder.current_size,
                    self._config.target_patch_space_size
                )
        
        # Construir patch space final
        patch_space = self._builder.build()
        self._stats.patch_space_size = patch_space.shape[0]
        logger.info("Patch space construido: %d parches", self._stats.patch_space_size)
        
        # Etapa 5: Submuestreo global
        logger.info("Etapa 5: Submuestreando...")
        X = self._subsampler.subsample(patch_space)
        self._stats.subsample_size = X.shape[0]
        logger.info("Submuestra: %d puntos", self._stats.subsample_size)
        
        # Etapa 6: Filtrado por densidad + denoising
        logger.info("Etapa 6: Filtrado por densidad y denoising...")
        X_pk = self._density_filter.transform(X)
        self._stats.final_size = X_pk.shape[0]
        
        logger.info(
            "Pipeline completado. Estadísticas finales:\n"
            "  - Imágenes procesadas: %d\n"
            "  - Parches extraídos: %d\n"
            "  - Parches tras D-norm: %d\n"
            "  - Patch space: %d\n"
            "  - Submuestra: %d\n"
            "  - X(p,k) final: %d",
            self._stats.images_processed,
            self._stats.total_patches_extracted,
            self._stats.patches_after_dnorm,
            self._stats.patch_space_size,
            self._stats.subsample_size,
            self._stats.final_size
        )
        
        return X_pk
