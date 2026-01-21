"""
patches_tda - Librería para construcción del Patch Space y TDA

Pipeline para reproducir resultados de:
"On the Local Behavior of Spaces of Natural Images" (Carlsson et al., 2008)
"""

__version__ = "0.1.0"

# API Pública
from patches_tda.io.imc_image_loader import IMCImageLoader
from patches_tda.extraction.patch_extractor import PatchExtractor
from patches_tda.extraction.top_dnorm_selector import TopDNormSelector
from patches_tda.generators.polynomial_patch_generator import PolynomialPatchGenerator
from patches_tda.dnorm.dnorm_calculator import DNormCalculator
from patches_tda.space.patch_space_builder import PatchSpaceBuilder
from patches_tda.space.global_subsampler import GlobalSubsampler
from patches_tda.filtering.density_filter import DensityFilter
from patches_tda.pipeline.patch_space_pipeline import PatchSpacePipeline, PipelineConfig
from patches_tda.tda.witness_tda import WitnessTDA

__all__ = [
    "IMCImageLoader",
    "PatchExtractor",
    "TopDNormSelector",
    "PolynomialPatchGenerator",
    "DNormCalculator",
    "PatchSpaceBuilder",
    "GlobalSubsampler",
    "DensityFilter",
    "PatchSpacePipeline",
    "PipelineConfig",
    "WitnessTDA",
]
