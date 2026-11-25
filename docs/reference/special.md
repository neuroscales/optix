# Special functions (`optix.special`)

Quadrature, batched Newton solvers and KL divergences, written to avoid
cancellation. See [Probability distributions](../concepts/distributions.md#numerical-stability).

::: optix.special
    options:
      heading_level: 2
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - gauss_hermite
        - gauss_legendre
        - log_minus_digamma
        - stirling_residual
        - ratio_minus_log
        - fit_gamma_shape
        - inv_digamma
        - dirichlet_elog
        - fit_dirichlet
        - kl_gamma
        - kl_dirichlet
        - kl_normal
        - clipped_normal_moments
        - poisson_normal_mean
        - poisson_normal_logpdf
        - poisson_normal_ratio
        - bessel_j0
        - bessel_j1
        - bessel_j012
