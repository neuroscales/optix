"""Building blocks of convex optimisation: proximity operators, functions,
finite-difference operators and solvers.

Semi-public: these may change without notice. They are shared by the
regularised estimators (for example
[`Ikoma`][optix.deconv.Ikoma]).

- Proximity operators: `prox_box`, `prox_group_l2`, `project_group_l2`,
  `prox_schatten` (with `schatten_norm`), `project_schatten`,
  `prox_squared_l2`, `prox_shifted_poisson`, `prox_conjugate`; the
  iterative `prox_hessian_schatten` (with `HessianSchattenResult`).
- Functions pairing a value and a prox: `Box`, `GroupL2`, `Schatten`,
  `HessianSchatten`, `SquaredL2`, `ShiftedPoisson`; smooth functions
  (value, gradient, Lipschitz constant): the protocol `Smooth` and
  `PoissonGaussian`.
- Linear operators: the protocols `LinearOperator` (forward, adjoint,
  norm2) and `Circulant` (adds the Fourier symbols); `Identity`,
  `Convolution` (a transfer function), the finite differences `diff`,
  `diff_adjoint`, `Gradient` and `Hessian`; `as_circulant`.
- Solvers: `admm_fft`; `primal_dual` (with `primal_dual_step`).
"""

from ._admm import AdmmResult, Term, admm_fft
from ._diff import Gradient, Hessian, diff, diff_adjoint, rfft_frequencies
from ._functions import (
    Box,
    Function,
    GroupL2,
    PoissonGaussian,
    Schatten,
    ShiftedPoisson,
    Smooth,
    SquaredL2,
)
from ._hessian_schatten import (
    HessianSchatten,
    HessianSchattenResult,
    prox_hessian_schatten,
)
from ._operators import (
    Circulant,
    Convolution,
    Identity,
    LinearOperator,
    as_circulant,
)
from ._primal_dual import PrimalDualResult, primal_dual, primal_dual_step
from ._prox import (
    project_group_l2,
    project_schatten,
    prox_box,
    prox_conjugate,
    prox_group_l2,
    prox_schatten,
    prox_shifted_poisson,
    prox_squared_l2,
    schatten_norm,
)

__all__ = [
    "AdmmResult",
    "Box",
    "Circulant",
    "Convolution",
    "Function",
    "Gradient",
    "GroupL2",
    "Hessian",
    "HessianSchatten",
    "HessianSchattenResult",
    "Identity",
    "LinearOperator",
    "PoissonGaussian",
    "PrimalDualResult",
    "Schatten",
    "ShiftedPoisson",
    "Smooth",
    "SquaredL2",
    "Term",
    "admm_fft",
    "as_circulant",
    "diff",
    "diff_adjoint",
    "primal_dual",
    "primal_dual_step",
    "project_group_l2",
    "project_schatten",
    "prox_box",
    "prox_conjugate",
    "prox_group_l2",
    "prox_hessian_schatten",
    "prox_schatten",
    "prox_shifted_poisson",
    "prox_squared_l2",
    "rfft_frequencies",
    "schatten_norm",
]
