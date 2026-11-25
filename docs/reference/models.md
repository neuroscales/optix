# Forward models (`optix.models`)

Simulators of the blur, camera and shading models that return ground truth as
estimates, and [phantoms](#phantoms) to feed them. See
[Image formation and noise](../concepts/image-formation.md).

::: optix.models
    options:
      heading_level: 2
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gaussian_psf
        - simulate_blur
        - random_sensor
        - simulate_sensor
        - random_shading
        - simulate_shading
        - simulate_tiles

## Phantoms

`optix.models.phantoms` draws ground-truth specimens: latent images
$\mathbf{x} \geq 0$ in expected photoelectrons per pixel, to pass through
the simulators above. A scene is a short composition:

```python
from optix.models import phantoms, random_sensor, random_shading
from optix.models import simulate_sensor, simulate_shading

x = phantoms.cells((40, 64, 64), seed=0) + 20  # 40 frames of cells
shading = random_shading((64, 64), seed=1)
y = simulate_sensor(simulate_shading(x, shading), random_sensor((64, 64)))
```

::: optix.models.phantoms
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - cells
        - beads
        - filaments
        - background
        - stripes
        - ramp
