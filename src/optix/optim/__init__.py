"""Building blocks of convex optimisation: proximity operators, functions,
finite-difference operators and solvers.

Semi-public: these may change without notice. They are shared by the
regularised estimators (for example
[`Ikoma`][optix.deconv.Ikoma]).

- Proximity operators: `prox_box`, `prox_group_l2`, `project_group_l2`,
  `prox_schatten` (with `schatten_norm`), `prox_squared_l2`,
  `prox_shifted_poisson`, `prox_conjugate`.
- Functions pairing a value and a prox: `Box`, `GroupL2`, `Schatten`,
  `SquaredL2`, `ShiftedPoisson`.
- Linear operators: the protocols `LinearOperator` (forward, adjoint,
  norm2) and `Circulant` (adds the Fourier symbols); `Identity`,
  `Convolution` (a transfer function), the finite differences `diff`,
  `diff_adjoint`, `Gradient` and `Hessian`; `as_circulant`.
- Solvers: `admm_fft`.
"""

from ._admm import AdmmResult, Term, admm_fft
from ._diff import Gradient, Hessian, diff, diff_adjoint, rfft_frequencies
from ._functions import (
    Box,
    Function,
    GroupL2,
    Schatten,
    ShiftedPoisson,
    SquaredL2,
)
from ._operators import (
    Circulant,
    Convolution,
    Identity,
    LinearOperator,
    as_circulant,
)
from ._prox import (
    project_group_l2,
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
    "Identity",
    "LinearOperator",
    "Schatten",
    "ShiftedPoisson",
    "SquaredL2",
    "Term",
    "admm_fft",
    "as_circulant",
    "diff",
    "diff_adjoint",
    "project_group_l2",
    "prox_box",
    "prox_conjugate",
    "prox_group_l2",
    "prox_schatten",
    "prox_shifted_poisson",
    "prox_squared_l2",
    "rfft_frequencies",
    "schatten_norm",
]
