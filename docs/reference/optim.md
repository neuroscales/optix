# Optimisation (`optix.optim`)

Proximity operators, finite-difference operators and ADMM and primal–dual solvers, the building
blocks of the regularised estimators. See [Optimisation tools](../concepts/optimisation.md).

!!! warning "No stability promise"
    `optix.optim` is semi-public: it may change without notice.

## Proximity operators

::: optix.optim
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - prox_box
        - prox_group_l2
        - project_group_l2
        - prox_schatten
        - schatten_norm
        - project_schatten
        - prox_hessian_schatten
        - HessianSchattenResult
        - prox_squared_l2
        - prox_shifted_poisson
        - prox_conjugate

## Functions

::: optix.optim
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - Function
        - Box
        - GroupL2
        - Schatten
        - HessianSchatten
        - SquaredL2
        - ShiftedPoisson
        - Smooth
        - PoissonGaussian

## Linear operators

::: optix.optim
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - LinearOperator
        - Circulant
        - Identity
        - Convolution
        - as_circulant

## Difference operators

::: optix.optim
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - diff
        - diff_adjoint
        - Gradient
        - Hessian
        - rfft_frequencies

## Solver

::: optix.optim
    options:
      heading_level: 3
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - admm_fft
        - Term
        - AdmmResult
        - primal_dual
        - primal_dual_step
        - PrimalDualResult
