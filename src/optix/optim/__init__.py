"""Building blocks of convex optimisation: proximity operators, functions,
finite-difference operators and solvers.

Semi-public: these may change without notice. They are shared by the
regularised estimators (for example
[`Ikoma`][optix.deconv.Ikoma]).

- Proximity operators: `prox_box`, `prox_group_l2`, `project_group_l2`,
  `prox_squared_l2`, `prox_shifted_poisson`, `prox_conjugate`.
- Functions pairing a value and a prox: `Box`, `GroupL2`, `SquaredL2`,
  `ShiftedPoisson`.
- Linear operators with forward, adjoint and Fourier symbol: `diff`,
  `diff_adjoint`, `Gradient`, `Hessian`.
- Solvers: `admm_fft`.
"""

from ._admm import AdmmResult, Term, admm_fft
from ._diff import Gradient, Hessian, diff, diff_adjoint, rfft_frequencies
from ._functions import Box, Function, GroupL2, ShiftedPoisson, SquaredL2
from ._prox import (
    project_group_l2,
    prox_box,
    prox_conjugate,
    prox_group_l2,
    prox_shifted_poisson,
    prox_squared_l2,
)

__all__ = [
    "AdmmResult",
    "Box",
    "Function",
    "Gradient",
    "GroupL2",
    "Hessian",
    "ShiftedPoisson",
    "SquaredL2",
    "Term",
    "admm_fft",
    "diff",
    "diff_adjoint",
    "project_group_l2",
    "prox_box",
    "prox_conjugate",
    "prox_group_l2",
    "prox_shifted_poisson",
    "prox_squared_l2",
    "rfft_frequencies",
]
