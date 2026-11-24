---
icon: fontawesome/solid/lightbulb
---

# Concepts

These pages explain the ideas behind optix, without going through every
function. They are meant to be read once, in order, and then used as a reference
when an algorithm page assumes something. Each page runs on the same few
objects, and each example can be pasted into a Python session.

- [Axes and layouts](axes-and-layouts.md) explains how an array is read as a
  pile of pictures, with axes that are *spatial*, *sample* or *group*. It
  covers OME-Zarr stacks, light-sheet volumes and the streaming of arrays that
  do not fit in memory.
- [Image formation and noise](image-formation.md) says what a pixel value
  means: the camera model, photon transfer, quantisation, and the shading model
  $\mathbf{y}_n = \mathbf{f} \odot (\mathbf{x}_n + b_n) + \mathbf{d}$. It defines
  the conventions that results follow.
- [Estimators](estimators.md) describes the `fit` / `apply` contract, faithful
  mode and extensions, starting one estimator from another, and the choice of
  device, precision and memory.
- [Probability distributions](distributions.md) presents the Gamma, Normal,
  Dirichlet and Categorical families: their shapes, sufficient statistics and
  conjugate updates, and what keeps them numerically safe.
- [Optimisation tools](optimisation.md) presents `optix.optim`: proximity
  operators, finite-difference operators with their Fourier symbols, and the
  ADMM solver that regularised estimators such as
  [Ikoma](../algorithms/deconv/ikoma.md) are built on.
- [Notation](notation.md) states the typographic rules and lists every
  symbol, relates the flat field, dark field and baseline to the physics of the
  microscope, and shows how algorithm pages give the notation of their paper.
