# patches_tda API Reference

> Pipeline para construcción del Patch Space y TDA basado en  
> *"On the Local Behavior of Spaces of Natural Images"* (Carlsson et al., 2008)

---

## Instalación y Uso

```python
import sys
sys.path.insert(0, "src")  # O agregar al PYTHONPATH

from patches_tda import (
    IMCImageLoader,
    PatchExtractor,
    DNormCalculator,
    TopDNormSelector,
    PatchSpaceBuilder,
    GlobalSubsampler,
    DensityFilter,
    PatchSpacePipeline,
    PipelineConfig,
)
```

---

## Módulo I/O

### `IMCImageLoader`

Carga imágenes Van Hateren en formato `.imc` (binario uint16 big-endian, 1024×1536).

#### Constructor

```python
IMCImageLoader(data_dir: Path | str)
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `data_dir` | `Path \| str` | Directorio con archivos `.imc` |

#### Métodos

| Método | Retorno | Descripción |
|--------|---------|-------------|
| `load(filename)` | `np.ndarray` | Carga imagen como float64 (1024, 1536) |
| `list_images()` | `list[str]` | Lista archivos .imc en el directorio |
| `iter_images(limit=None)` | `Iterator` | Itera sobre imágenes (filename, array) |

#### Ejemplo

```python
from pathlib import Path
from patches_tda import IMCImageLoader

loader = IMCImageLoader(Path("vanhateren_imc"))

# Listar imágenes
print(loader.list_images()[:5])
# ['imk00001.imc', 'imk00002.imc', ...]

# Cargar una imagen
image = loader.load("imk00001.imc")
print(image.shape, image.dtype)
# (1024, 1536) float64

# Iterar sobre 10 imágenes
for filename, img in loader.iter_images(limit=10):
    print(f"{filename}: min={img.min():.0f}, max={img.max():.0f}")
```

---

## Módulo Extraction

### `PatchExtractor`

Extrae parches cuadrados m×m de una imagen, aplica log-intensidad y centrado.

#### Constructor

```python
PatchExtractor(
    patch_size: int = 3,
    n_patches: int = 5000,
    rng: np.random.Generator | None = None
)
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `patch_size` | `int` | 3 | Tamaño del parche (m) |
| `n_patches` | `int` | 5000 | Parches a extraer por imagen (Y) |
| `rng` | `Generator` | None | Generador para reproducibilidad |

#### Propiedades

| Propiedad | Tipo | Descripción |
|-----------|------|-------------|
| `patch_size` | `int` | Tamaño m |
| `n_patches` | `int` | Número de parches |
| `patch_dim` | `int` | Dimensión = m² (e.g., 9 para 3×3) |

#### Métodos

| Método | Retorno | Descripción |
|--------|---------|-------------|
| `extract(image)` | `np.ndarray` | Retorna (n_patches, m²) parches procesados |

#### Pipeline de extracción

1. Muestrear posiciones aleatorias (evitando bordes)
2. Extraer parches m×m
3. Aplicar `log(1 + x)` (log-intensidad)
4. Centrar cada parche (restar su media)
5. Vectorizar en row-major → ℝ^{m²}

#### Ejemplo

```python
import numpy as np
from patches_tda import PatchExtractor

# Reproducible
rng = np.random.default_rng(42)
extractor = PatchExtractor(patch_size=3, n_patches=1000, rng=rng)

# Imagen de prueba
image = np.random.rand(1024, 1536) * 10000

patches = extractor.extract(image)
print(patches.shape)  # (1000, 9)
print(patches.mean(axis=1).max())  # ≈ 0 (centrados)
```

---

### `TopDNormSelector`

Selecciona el top q% de parches con mayor D-norm (contraste).

#### Constructor

```python
TopDNormSelector(
    dnorm_calculator: DNormCalculator,
    top_fraction: float = 0.20
)
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `dnorm_calculator` | `DNormCalculator` | — | Calculador de D-norm |
| `top_fraction` | `float` | 0.20 | Fracción a conservar (q) |

#### Propiedades

| Propiedad | Tipo | Descripción |
|-----------|------|-------------|
| `top_fraction` | `float` | Fracción q |
| `last_dnorms` | `np.ndarray \| None` | D-norms de última selección |

#### Ejemplo

```python
from patches_tda import DNormCalculator, TopDNormSelector, PatchExtractor

calc = DNormCalculator(patch_size=3)
selector = TopDNormSelector(calc, top_fraction=0.20)

extractor = PatchExtractor(patch_size=3, n_patches=5000)
patches = extractor.extract(image)

# Seleccionar top 20% por contraste
selected = selector.select(patches)
print(selected.shape)  # (1000, 9) — 20% de 5000
```

---

## Módulo D-Norm

### `DNormCalculator`

Calcula la D-norm (norma de contraste) definida como `‖x‖_D = √(xᵀ D x)`.

La matriz D se construye como D = AᵀA donde A contiene diferencias entre píxeles adyacentes (horizontales y verticales).

#### Constructor

```python
DNormCalculator(patch_size: int = 3)
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `patch_size` | `int` | 3 | Tamaño del parche m |

#### Propiedades

| Propiedad | Tipo | Descripción |
|-----------|------|-------------|
| `patch_size` | `int` | Tamaño m |
| `D_matrix` | `np.ndarray` | Matriz D de shape (m², m²) |

#### Métodos

| Método | Retorno | Descripción |
|--------|---------|-------------|
| `compute(patches)` | `np.ndarray` | D-norms de shape (N,) |

#### Ejemplo

```python
import numpy as np
from patches_tda import DNormCalculator

calc = DNormCalculator(patch_size=3)
print(calc.D_matrix.shape)  # (9, 9)

# Parche uniforme (bajo contraste)
uniform = np.ones((1, 9)) * 5.0
print(calc.compute(uniform))  # [0.0]

# Parche con gradiente (alto contraste)
gradient = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
print(calc.compute(gradient))  # [~6.93]
```

---

## Módulo Space

### `PatchSpaceBuilder`

Acumula parches de múltiples imágenes hasta alcanzar Z_target.

#### Constructor

```python
PatchSpaceBuilder(
    target_size: int = 4_000_000,
    patch_dim: int = 9
)
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `target_size` | `int` | 4×10⁶ | Tamaño objetivo (Z_target) |
| `patch_dim` | `int` | 9 | Dimensión de cada parche |

#### Propiedades

| Propiedad | Tipo | Descripción |
|-----------|------|-------------|
| `target_size` | `int` | Z_target |
| `current_size` | `int` | Parches acumulados |
| `is_full` | `bool` | True si alcanzó target |

#### Métodos

| Método | Retorno | Descripción |
|--------|---------|-------------|
| `add_patches(patches)` | `int` | Agrega parches, retorna cantidad agregada |
| `build()` | `np.ndarray` | Retorna patch space 𝓜 |
| `reset()` | `None` | Reinicia el builder |
| `save(path)` | `None` | Guarda a .npy |
| `load(path)` | `None` | Carga desde .npy |

#### Ejemplo

```python
from patches_tda import PatchSpaceBuilder

builder = PatchSpaceBuilder(target_size=100_000, patch_dim=9)

# Simular procesamiento de imágenes
for i in range(50):
    fake_patches = np.random.randn(2500, 9)  # 2500 parches/imagen
    n_added = builder.add_patches(fake_patches)
    
    if builder.is_full:
        print(f"Lleno en imagen {i+1}")
        break

patch_space = builder.build()
print(patch_space.shape)  # (100000, 9)

# Persistencia
builder.save("patch_space.npy")
```

---

### `GlobalSubsampler`

Submuestreo aleatorio de N puntos del patch space.

#### Constructor

```python
GlobalSubsampler(
    n_samples: int = 50_000,
    rng: np.random.Generator | None = None
)
```

#### Ejemplo

```python
from patches_tda import GlobalSubsampler

sampler = GlobalSubsampler(n_samples=50_000, rng=np.random.default_rng(42))

X = sampler.subsample(patch_space)  # patch_space: (4_000_000, 9)
print(X.shape)  # (50000, 9)
```

---

## Módulo Filtering

### `DensityFilter`

Filtrado por densidad + denoising para obtener X(p,k).

#### Constructor

```python
DensityFilter(
    k_neighbors: int = 15,
    top_density_fraction: float = 0.30,
    denoise_iterations: int = 2
)
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `k_neighbors` | `int` | 15 | Vecinos para densidad/denoising (k) |
| `top_density_fraction` | `float` | 0.30 | Fracción densa a conservar (p) |
| `denoise_iterations` | `int` | 2 | Iteraciones de denoising |

#### Métodos

| Método | Retorno | Descripción |
|--------|---------|-------------|
| `estimate_density(X)` | `np.ndarray` | Densidades (N,) |
| `filter_by_density(X)` | `np.ndarray` | Top p% más densos |
| `denoise(X)` | `np.ndarray` | Aplica denoising |
| `transform(X)` | `np.ndarray` | Pipeline completo → X(p,k) |

#### Pipeline de transform()

1. Estimar densidad local (1/distancia al k-ésimo vecino)
2. Conservar top p% más densos
3. Aplicar denoising (promedio de k vecinos) × iteraciones

#### Ejemplo

```python
from patches_tda import DensityFilter

filt = DensityFilter(
    k_neighbors=15,
    top_density_fraction=0.30,
    denoise_iterations=2
)

X_pk = filt.transform(X)  # X: (50000, 9)
print(X_pk.shape)  # (~15000, 9) — 30% de 50000
```

---

## Módulo Pipeline

### `PipelineConfig`

Dataclass con toda la configuración del pipeline.

```python
@dataclass
class PipelineConfig:
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
```

---

### `PatchSpacePipeline`

Orquesta el pipeline completo.

#### Flujo de ejecución

```
Imágenes .imc
    ↓ IMCImageLoader.iter_images()
Parches aleatorios (Y por imagen)
    ↓ PatchExtractor.extract()
Top q% por D-norm (por imagen)
    ↓ TopDNormSelector.select()
Patch Space 𝓜 (≈ Z_target)
    ↓ PatchSpaceBuilder.build()
Submuestreo global X (N)
    ↓ GlobalSubsampler.subsample()
Filtrado + denoising → X(p,k)
    ↓ DensityFilter.transform()
```

#### Ejemplo Completo

```python
from pathlib import Path
from patches_tda import PatchSpacePipeline, PipelineConfig

config = PipelineConfig(
    data_dir=Path("vanhateren_imc"),
    patch_size=3,
    patches_per_image=5000,
    top_dnorm_fraction=0.20,
    target_patch_space_size=1_000_000,  # 1M para prueba
    subsample_size=50_000,
    density_k=15,
    density_top_fraction=0.30,
    denoise_iterations=2,
    seed=42,
    max_images=100  # Limitar para prueba rápida
)

pipeline = PatchSpacePipeline(config)
X_pk = pipeline.run()

print(f"Resultado final: {X_pk.shape}")
print(pipeline.last_stats)
```

#### Estadísticas

```python
@dataclass
class PipelineStats:
    images_processed: int
    total_patches_extracted: int
    patches_after_dnorm: int
    patch_space_size: int
    subsample_size: int
    final_size: int
```

---

## Constantes del Formato .imc

```python
from patches_tda.io.imc_image_loader import (
    IMC_HEIGHT,  # 1024
    IMC_WIDTH,   # 1536
    IMC_EXPECTED_BYTES,  # 3145728
)
```
