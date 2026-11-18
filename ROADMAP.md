# optix — algorithm survey and roadmap

Sources: four literature surveys (flat-field/destriping, deconvolution/PSF, sensor/denoising,
other LSFM + OCT + bright-field). Citations marked **[U]** could not be verified (the sandbox
cannot reach Crossref/PMC) and must be checked before they appear in the docs.

---

## 1. Package organisation

Python modules form a tree, so one axis has to be primary. **Recommendation: organise by
*what is estimated*, and make the *assumptions* explicit through shared model components
and tags.** Users go looking for "flat-field correction", not for "low-rank + sparse"; and
comparing or chaining algorithms needs a common result type for each estimation target.

```
src/optix/
  _config.py        batch size / dtype / device / compile, as a context manager (not module globals)
  io/               lazy array access: frame batches, row tiles, nD blocks with halo, out= writers
  stats/            streaming per-pixel reducers: Welford moments, histograms / quantile sketches,
                    reservoir samples, order statistics over row tiles
  basis/            DCT bases + bending energy, B-splines, Zernike polynomials
  special/          Gauss–Hermite, gamma-shape / Dirichlet Newton solvers, KL terms
  fft/              fast padded sizes, (r)FFT convolution, halo bookkeeping
  models/           forward-model building blocks shared across tasks (the "assumptions"):
                    sensor (Poisson–Gaussian, per-pixel sCMOS), shading (multiplicative + additive),
                    blur (shift-invariant / spatially varying), attenuation (Beer–Lambert),
                    speckle (Rayleigh / K)
  sensor/           camera offset / gain / read noise, defective pixels, variance stabilisation
  shading/          flat field, dark field, per-frame baseline
  destripe/         stripe and shadow artefacts
  attenuation/      depth/illumination-axis decay
  background/       rolling ball, top-hat
  denoise/
  psf/              PSF models, bead extraction, phase retrieval
  deconv/
  fusion/           tile intensity harmonisation, multiview weighting (later: registration)
  modalities/       lightsheet/, oct/, brightfield/ — thin presets and pipelines that wire the
                    generic estimators together with modality defaults (axes, units, priors)
```

Joint models (such as the Gamma-mixture model) estimate several targets at once. They live
under their primary target (`shading/gmm.py`) and return the **shared result types** for
every target they estimate (`ShadingEstimate(flat, dark, baseline)`,
`SensorEstimate(offset, variance, gain)`, `Classification(Z)`, ...). Those types are how
algorithms get compared and how one initialises another: for example, BaSiC → `M` and `C`
for the GMM, or the GMM → `dark` and `flat` for RL deconvolution.

Each estimator also declares its assumptions as metadata (modality, noise model, needs many
frames, streaming class: pointwise / per-frame / halo / per-pixel-statistics / whole-stack).
The docs build an "assumption × target" table from that metadata.

### Conventions every estimator follows

- `fit(Y) -> self`, `apply(Y, out=None)`, results as dataclasses; `Y` is any array-like with
  `.shape` and slicing (numpy, zarr, tensor), only ever loaded batch by batch.
- `faithful=True` (default) reproduces the published algorithm, including its preprocessing
  and default hyper-parameters. Every extension is an explicit keyword argument that is off
  by default, documented in an "Extensions" section.
- Validation: compare against the reference implementation on stored fixtures (BaSiCPy,
  CIDRE MATLAB, DeconvolutionLab2, pyCUDAdecon), plus synthetic data drawn from
  `models/` with known ground truth.

---

## 2. Algorithms to implement

Priority: **P1** = first wave, **P2** = second wave, **P3** = later. The tier column is the
suggested delegation level under the escalation rules.

### Sensor / camera (`sensor/`)
| Algorithm | Reference | Estimates | Streaming | Prio | Tier |
|---|---|---|---|---|---|
| Prospective dark/flat calibration | standard practice | offset, flat | per-pixel stats | P1 | Low Opus |
| Photon-transfer curve (EMVA 1288) | EMVA 1288 r4.0 (2021) | gain, read noise, DSNU/PRNU | per-pixel stats | P1 | Low Opus |
| Pixel-wise sCMOS calibration | Huang et al., Nat Methods 2013, 10.1038/nmeth.2488 | per-pixel offset / variance / gain | per-pixel stats | P1 | Low Opus |
| Defective / hot / RTS pixel detection | generic (robust MAD) | mask | per-pixel stats | P1 | Low Opus |
| Poisson–Gaussian fit from data | Foi et al., IEEE TIP 2008, 10.1109/TIP.2008.2001399 | σ²(y) = a·y + b | frame subset | P1 | High Opus |
| Generalised Anscombe + exact unbiased inverse | Mäkitalo & Foi, IEEE TIP 2011 (10.1109/TIP.2010.2056693) and 2013 (10.1109/TIP.2012.2202675) | VST | pointwise | P1 | Low Opus |

### Shading: flat / dark / baseline (`shading/`)
| Algorithm | Reference | Model | Streaming | Prio | Tier |
|---|---|---|---|---|---|
| Mean/median illumination function | Singh et al., J Microsc 2014, 10.1111/jmi.12178 | multiplicative, smooth | running mean / sketch | P1 | Low Opus |
| **BaSiC** | Peng et al., Nat Commun 2017, 10.1038/ncomms14836 | I = S·I⁰ + D + bᵢ, low-rank + sparse, DCT, L1, ADMM | fit on a subsample, apply in batches | P1 | High Opus |
| **CIDRE** | Smith et al., Nat Methods 2015, 10.1038/nmeth.3323 | I = S·v + z, regularised energy on per-pixel order statistics | per-pixel sketch | P1 | High Opus |
| **Gamma-mixture VB model (ours)** | — | sensor + shading + tissue classes | frame batches, row tiles | P1 | High Opus + Fable review |
| Entropy minimisation | Likar et al., J Microsc 2000 [DOI U]; Tomažević et al. 2002 | polynomial shading | per frame | P3 | Low Opus |
| Kask (separate foreground/background profiles) | Kask et al., J Microsc 2016 [DOI U] | two profiles | per-pixel stats | P3 | Low Opus |

BaSiCPy 2.x is MIT-licensed and PyTorch-based, which makes it a good numerical reference for
the faithful mode. Check which options it added since the paper (`sort_intensity`, autotuning,
approximate fitting) [U]; those are candidate extensions, not faithful-mode defaults.

### LSFM artefacts (`destripe/`, `attenuation/`, `background/`, `fusion/`)
| Algorithm | Reference | Streaming | Prio | Tier |
|---|---|---|---|---|
| Wavelet–FFT destriping | Münch, Trtik, Marone, Stampanoni, Opt Express 2009, 10.1364/OE.17.008567 | per frame | P1 | Low Opus |
| PyStripe options (dual band, etc.) as extensions | Swaney et al. 2019 (Chung lab) [venue U] | per frame | P2 | Low Opus |
| VSNR variational destriping | Fehrenbach, Weiss, Lorenzo, IEEE TIP 2012 [DOI U] | per frame, iterative | P2 | High Opus |
| Variational destriping | Rottmayer et al., Opt Express 2025, 10.1364/OE.542868 | per frame | P3 | High Opus |
| Beer–Lambert attenuation along the sheet axis | e.g. Frantz et al., BOE 2024, 10.1364/BOE.531210 | per-column stats | P2 | Low Opus |
| Rolling ball / top-hat background | Sternberg 1983, 10.1109/MC.1983.1654163 | halo | P2 | Low Opus |
| Tile intensity harmonisation (gain/offset from overlaps) | BigStitcher, Hörl et al. 2019, 10.1038/s41592-019-0501-0 | overlap stats | P2 | Low Opus |
| Content-based multiview fusion weights | Preibisch et al. 2014 (+ earlier ref U) | halo | P3 | Low Opus |

### PSF (`psf/`)
| Algorithm | Reference | Prio | Tier |
|---|---|---|---|
| Gibson–Lanni scalar PSF, fast Bessel-series evaluation | Gibson & Lanni, JOSA A 1991, 10.1364/JOSAA.8.001601; Li, Xue, Blu, JOSA A 2017, 10.1364/JOSAA.34.001029 | P1 | High Opus |
| LSFM effective PSF = detection PSF × Gaussian-beam sheet profile | standard | P1 | Low Opus |
| Bead detection, alignment and averaging | standard | P1 | Low Opus |
| Richards–Wolf vectorial PSF (high NA) | Richards & Wolf 1959, 10.1098/rspa.1959.0200 | P2 | High Opus |
| Pupil phase retrieval + Zernike fit | Hanser et al., J Microsc 2004 [DOI U] | P2 | High Opus |

### Deconvolution (`deconv/`)
All of these use block-wise processing with a halo of at least the PSF radius, and FFT sizes
padded to 2ᵃ3ᵇ5ᶜ.

| Algorithm | Reference | Prio | Tier |
|---|---|---|---|
| Richardson–Lucy | Richardson 1972, 10.1364/JOSA.62.000055; Lucy 1974, 10.1086/111605 | P1 | Low Opus |
| Biggs–Andrews acceleration | Appl Opt 1997, 10.1364/AO.36.001766 | P1 | Low Opus |
| Extensions: camera offset/gain, background term, Poisson–Gaussian (shifted Poisson), damping | — | P1 | Low Opus |
| Unmatched Wiener–Butterworth back-projector | Guo et al., Nat Biotech 2020, 10.1038/s41587-020-0560-x | P1 | Low Opus |
| Wiener / Tikhonov | classical | P2 | Low Opus |
| RL-TV | Dey et al., Microsc Res Tech 2006, 10.1002/jemt.20294 | P2 | Low Opus |
| Spatially varying PSF (per-block, then interpolated) | Nagy & O'Leary 1998, 10.1137/S106482759528507X; Denis et al. 2015, 10.1007/s11263-015-0817-x | P2 | High Opus |
| Multiview RL | Preibisch et al., Nat Methods 2014, 10.1038/nmeth.2929 | P3 (needs registration) | High Opus |
| Primal–dual / ADMM with Poisson–Gaussian likelihood | various [U] | P3 | High Opus |

### Denoising (`denoise/`)
| Algorithm | Reference | Prio | Tier |
|---|---|---|---|
| PURE-LET (mixed Poisson–Gaussian) | Luisier, Blu, Unser, IEEE TIP 2011, 10.1109/TIP.2010.2073477 | P2 | High Opus |
| TV / Hessian regularised denoising (shared prox with RL-TV) | ROF 1992 [U]; Huang et al. Nat Biotech 2018, 10.1038/nbt.4115 | P2 | Low Opus |
| Non-local means (on top of the VST) | Buades et al. 2005 [U] | P2 | Low Opus |
| BM3D / BM4D | Dabov 2007, Maggioni 2013 [U] | P3 | High Opus |
| NCS (sCMOS) | Liu et al., Nat Methods 2017, 10.1038/nmeth.4379 | P3 | High Opus |
| ACsN (sCMOS) | Mandracchia et al., Nat Commun 2020, 10.1038/s41467-019-13841-8 | P3 | High Opus |
| Self-supervised DL (Noise2Void, Noise2Self, DeepCAD) | — | out of scope for now | — |

### Other modalities (`modalities/oct`, `modalities/brightfield`), P3
- OCT: depth-resolved attenuation (Vermeer et al., BOE 2014, 10.1364/BOE.5.000322) and
  compensation (Girard et al., IOVS 2011, 10.1167/iovs.10-6925). Both are cumulative sums
  along A-scans and share code with LSFM attenuation. Also speckle statistics models
  (Schmitt et al., JBO 1999, 10.1117/1.429925) with despeckling built on them, and numerical
  dispersion compensation (Wojtkowski et al., Opt Express 2004, 10.1364/OPEX.12.002404).
  Dispersion compensation needs raw spectra.
- Bright-field: colour deconvolution (Ruifrok & Johnston 2001), I₀/optical-density handling,
  TIE phase retrieval (Paganin et al., J Microsc 2002, 10.1046/j.1365-2818.2002.01010.x).
  Fourier ptychography is low priority.

---

## 3. Phasing

0. **Infrastructure** (blocks everything): `_config`, `io` (batches, row tiles, halo blocks,
   writers), `stats` reducers, `basis`, `special`, plus synthetic-data simulators from
   `models/`. Getting the API right is architectural work, so it suits *Fable — Plan Only*,
   with execution by *High-Effort Opus*.
1. **Sensor + shading**: calibration, PTC, defective pixels, Foi fit, GAT, Singh, BaSiC,
   CIDRE, and the GMM port. These are closest to current research, and BaSiC and CIDRE are
   the natural baselines and initialisers for the GMM.
2. **LSFM artefacts**: Münch destriping, Beer–Lambert attenuation, rolling ball, tile
   harmonisation.
3. **PSF + deconvolution**: block-wise RL + Biggs–Andrews + extensions, Gibson–Lanni + sheet,
   beads, Wiener–Butterworth, then RL-TV, Wiener, spatially varying PSF.
4. **Denoising**: PURE-LET, TV/Hessian, NLM, then BM3D, NCS, ACsN.
5. **Multiview, OCT, bright-field.**

---

## 4. Notes for porting the Gamma-mixture model

- The global-option helpers are inconsistent: `_dtype()`/`_device()` write `_DTYPE`/`_DEVICE`,
  but `_backend()` reads `DTYPE`/`DEVICE`, so setting them has no effect. Replace them with a
  single `optix.config` context manager.
- `_frame_batches`, `_row_tiles`, `_store`, `dct_basis`, `bending_weights`,
  `fit_modulation`, `gauss_hermite`, `fit_gamma_shape`, `fit_dirichlet` and `kl_dirichlet`
  are generic. They belong in `io/`, `basis/` and `special/`, where BaSiC (DCT shading),
  CIDRE and the calibration code will reuse them.
- `Parameters(C, V, M, gain)` maps directly onto the shared `SensorEstimate` and
  `ShadingEstimate` types. The `init` method can then accept either a data-driven start
  (current behaviour) or a BaSiC/CIDRE/calibration result.

---

## 5. Documentation

- One page per algorithm, with these sections: *Overview* → *Image formation* (optics) →
  *Probabilistic model* → *Inference / derivation* → *Algorithm* (pseudocode) →
  *Faithful mode and extensions* → *Memory and streaming* → *Usage* → *References*. Let
  each section's length follow what the topic needs.
- A "Concepts" section with the shared models (sensor noise, shading, blur, attenuation). The
  algorithm pages link to it instead of re-deriving the basics.
- Writing is done by Sonnet agents; derivations and equations are checked by Opus agents
  before merge. Style references: the scikit-image and SciPy user guides, the Stan User's
  Guide, PyMC's example gallery.
- All [U] citations get verified at writing time, against the publisher page.
