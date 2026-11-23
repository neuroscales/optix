"""ADMM with circulant operators, solved in the Fourier domain."""

from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

import torch
from torch import Tensor

from ._functions import Function
from ._operators import Circulant, Convolution, Identity, as_circulant

__all__ = ["AdmmResult", "Term", "admm_fft"]


class Term(NamedTuple):
    """One term f(A x) of an [`admm_fft`][optix.optim.admm_fft] objective.

    Attributes
    ----------
    operator : Circulant, Tensor or None
        The operator A, a [`Circulant`][optix.optim.Circulant] (for
        example [`Hessian`][optix.optim.Hessian] or
        [`Convolution`][optix.optim.Convolution]). Two shorthands are
        accepted: None for the [`Identity`][optix.optim.Identity], and a
        complex tensor for the `Convolution` with that transfer function
        on the half spectrum of `torch.fft.rfftn` (e.g. the `otf` of
        [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution]).
    function : Function
        f, applied to the split variable z = A x of shape `(C, *N)`
        (C = 1 for the identity and a convolution).
    region : tuple of slice, optional
        Restricts f to `z[:, *region]`; f is 0 elsewhere, so z = A x + u
        there. None: the whole grid.
    """

    operator: Circulant | Tensor | None
    function: Function
    region: tuple[slice, ...] | None = None


class AdmmResult(NamedTuple):
    """Result of [`admm_fft`][optix.optim.admm_fft].

    Attributes
    ----------
    x : Tensor
        Last iterate, `(*N)`.
    splits : list of Tensor
        Split variables zₖ, `(Cₖ, *N)`.
    duals : list of Tensor
        Scaled dual variables uₖ, `(Cₖ, *N)`.
    n_iter : int
        Number of iterations.
    change : float
        Last relative change ‖xᵏ⁺¹ − xᵏ‖₂ / ‖xᵏ‖₂ (inf if not measured).
    primal_residual : float
        Last primal residual ‖r‖₂ = (∑ₖ ‖Aₖ x − zₖ‖²)^½ (Boyd et al.
        2011, §3.3).
    dual_residual : float
        Last dual residual ‖s‖₂ (Boyd et al. 2011, §3.3): ρ‖∑ₖ Aₖᵀ(zₖ −
        zₖ_prev)‖ = ρ‖∑ₖ Aₖᵀ uₖ‖ for `order="x-first"`, ρ‖A(x − x_prev)‖
        for `order="z-first"` (inf if not measured).
    """

    x: Tensor
    splits: list[Tensor]
    duals: list[Tensor]
    n_iter: int
    change: float
    primal_residual: float = float("inf")
    dual_residual: float = float("inf")


def _symbols(
    operator: Circulant,
    grid: tuple[int, ...],
    dtype: torch.dtype,
    device: object,
) -> list[Tensor] | None:
    """Transfer functions of the components; None for the identity."""
    if isinstance(operator, Identity):
        return None
    return list(operator.symbols(grid, dtype=dtype, device=device))


def _norm(x: Tensor) -> float:
    return float(torch.linalg.vector_norm(x, dtype=torch.float64))


def admm_fft(
    terms: Sequence[Term],
    grid: Sequence[int],
    *,
    rho: float,
    x0: Tensor | None = None,
    order: Literal["x-first", "z-first"] = "x-first",
    max_iter: int = 150,
    tol: float | None = 1e-3,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trace: dict | None = None,
    callback: Callable[[int, Tensor], object] | None = None,
) -> AdmmResult:
    """Minimise ∑ₖ fₖ(Aₖ x) by ADMM, with every Aₖ circulant on a grid.

    Each term is split, zₖ = Aₖ x, and the scaled augmented Lagrangian

        ∑ₖ fₖ(zₖ) + (ρ/2) ‖Aₖ x − zₖ + uₖ‖²

    is minimised alternately over x and the zₖ, with the dual update
    uₖ ← uₖ + Aₖ x − zₖ (Boyd et al. 2011, §3.1.1). The x-update is the
    least-squares solve

        x = (∑ₖ AₖᵀAₖ)⁻¹ ∑ₖ Aₖᵀ(zₖ − uₖ),

    exact in the Fourier domain because every Aₖ is diagonalised by the
    DFT (Ikoma et al. 2018, Eq. S16; Lefkimmiatis & Unser 2013, Eq. 42);
    ∑ₖ |Âₖ|² must be positive at every frequency (an identity term
    ensures it). The zₖ-update is zₖ = prox_{fₖ/ρ}(Aₖ x + uₖ) on the
    term's region and zₖ = Aₖ x + uₖ outside it.

    - `order="x-first"`: x, then z, then u, from z = u = 0 (Ikoma et
      al. 2018, Eqs. S9–S15).
    - `order="z-first"`: z from (x, u), then x, then u, from x = `x0`
      and u = 0 (Lefkimmiatis & Unser 2013, Eqs. 18a–c).

    Iterations stop when ‖xᵏ⁺¹ − xᵏ‖₂ ≤ `tol`·‖xᵏ‖₂ (tested once
    ‖xᵏ‖₂ > 0) or after `max_iter` iterations. The norms of x are
    computed in float64.

    !!! warning "A small change is not convergence"
        The relative-change test is the stopping rule of Ikoma et al.
        (2018) and Lefkimmiatis & Unser (2013). ADMM can move slowly
        while still far from the minimiser, so a stop by this test does
        not certify convergence. The primal and dual residuals of the
        last iterate (Boyd et al. 2011, §3.3) are returned in
        [`AdmmResult`][optix.optim.AdmmResult]; both tend to 0 at a
        solution.

    !!! references "References"
        1. Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. *Distributed
           optimization and statistical learning via the alternating
           direction method of multipliers.* Found Trends Mach Learn
           **3**(1):1-122, 2011.
           [doi:10.1561/2200000016](https://doi.org/10.1561/2200000016)
        2. Ikoma H, Broxton M, Kudo T, Wetzstein G. *A convex 3D
           deconvolution algorithm for low photon count fluorescence
           imaging.* Sci Rep **8**:11489, 2018.
           [doi:10.1038/s41598-018-29768-x](https://doi.org/10.1038/s41598-018-29768-x)
        3. Lefkimmiatis S, Unser M. *Poisson image reconstruction with
           Hessian Schatten-norm regularization.* IEEE Trans Image
           Process **22**(11):4314-4327, 2013.
           [doi:10.1109/TIP.2013.2271852](https://doi.org/10.1109/TIP.2013.2271852)

        ??? quote "BibTeX"
            ```bibtex
            @article{boyd2011distributed,
              author  = {Boyd, Stephen and Parikh, Neal and Chu, Eric and
                         Peleato, Borja and Eckstein, Jonathan},
              title   = {Distributed Optimization and Statistical
                         Learning via the Alternating Direction Method of
                         Multipliers},
              journal = {Foundations and Trends in Machine Learning},
              volume  = {3},
              number  = {1},
              pages   = {1--122},
              year    = {2011},
              doi     = {10.1561/2200000016}
            }
            @article{ikoma2018convex,
              author  = {Ikoma, Hayato and Broxton, Michael and Kudo,
                         Takamasa and Wetzstein, Gordon},
              title   = {A convex {3D} deconvolution algorithm for low
                         photon count fluorescence imaging},
              journal = {Scientific Reports},
              volume  = {8},
              pages   = {11489},
              year    = {2018},
              doi     = {10.1038/s41598-018-29768-x}
            }
            @article{lefkimmiatis2013poisson,
              author  = {Lefkimmiatis, Stamatios and Unser, Michael},
              title   = {Poisson Image Reconstruction With {Hessian}
                         {Schatten}-Norm Regularization},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22},
              number  = {11},
              pages   = {4314--4327},
              year    = {2013},
              doi     = {10.1109/TIP.2013.2271852}
            }
            ```

    Parameters
    ----------
    terms : sequence of Term
        The terms fₖ(Aₖ x).
    grid : sequence of int
        Periodic grid N of x.
    rho : float
        Penalty parameter ρ > 0.
    x0 : Tensor, optional
        Starting x, `(*N)` (used by `order="z-first"`; with
        `order="x-first"` the first x is computed from z = u = 0).
        Default: zeros.
    order : {"x-first", "z-first"}
        Update order (see above).
    max_iter : int
        Maximum number of iterations.
    tol : float, optional
        Relative-change tolerance; None runs `max_iter` iterations.
    dtype, device : optional
        Backend. Default: those of `x0`, else of the first transfer
        function in `terms`, else the default dtype on the CPU.
    trace : dict, optional
        Filled during the run (a callback can read it) with
        per-iteration lists: `"change"` (relative change of x, nan when
        not measured), `"residual"` (primal residual), `"dual_residual"`
        (see [`AdmmResult`][optix.optim.AdmmResult]) and `"objective"`
        (∑ₖ fₖ(zₖ) at the split variables, in float64; it equals the
        objective at x once the primal residual is 0). The last two cost
        extra FFTs and reductions and are only computed with a trace.
    callback : callable, optional
        `callback(k, x)` after each iteration.

    Returns
    -------
    AdmmResult
    """
    grid = tuple(int(n) for n in grid)
    ndim = len(grid)
    dims = tuple(range(-ndim, 0))
    if not rho > 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if order not in ("x-first", "z-first"):
        raise ValueError(f"order must be 'x-first' or 'z-first': {order!r}")
    operators = [as_circulant(term.operator, grid) for term in terms]
    if dtype is None or device is None:
        like = x0
        if like is None:
            like = next(
                (
                    op.transfer
                    for op in operators
                    if isinstance(op, Convolution)
                ),
                None,
            )
        if dtype is None:
            dtype = (
                torch.get_default_dtype()
                if like is None
                else (like.real.dtype if like.is_complex() else like.dtype)
            )
        if device is None:
            device = None if like is None else like.device
    t = 1.0 / float(rho)
    symbols = [_symbols(op, grid, dtype, device) for op in operators]
    half = (*grid[:-1], grid[-1] // 2 + 1)
    den = torch.zeros(half, dtype=dtype, device=device)
    for sym in symbols:
        if sym is None:
            den += 1
        else:
            for s in sym:
                den += s.real.square() + s.imag.square()
    if not bool((den > 0).all()):
        raise ValueError(
            "∑ₖ |Aₖ|² vanishes at some frequency: add an identity term"
        )
    inv_den = 1 / den
    del den

    def apply(sym: list[Tensor] | None, spec: Tensor, x: Tensor) -> Tensor:
        """Aₖ x, `(C, *N)`, a new tensor."""
        if sym is None:
            return x.clone()[None]
        out = x.new_empty((len(sym), *grid))
        for c, s in enumerate(sym):
            out[c] = torch.fft.irfftn(s * spec, s=grid, dim=dims)
        return out

    def adjoint_sum(fields: list[Tensor]) -> Tensor:
        """∑ₖ Aₖᵀ wₖ, `(*N)`."""
        acc = None
        for sym, w in zip(symbols, fields, strict=True):
            for c in range(w.shape[0]):
                r = torch.fft.rfftn(w[c], dim=dims)
                if sym is not None:
                    r *= sym[c].conj()
                acc = r if acc is None else acc.add_(r)
                del r
        return torch.fft.irfftn(acc, s=grid, dim=dims)

    def solve(splits: list[Tensor], duals: list[Tensor]) -> Tensor:
        rhs = None
        for sym, z, u in zip(symbols, splits, duals, strict=True):
            for c in range(z.shape[0]):
                r = torch.fft.rfftn(z[c] - u[c], dim=dims)
                if sym is not None:
                    r *= sym[c].conj()
                rhs = r if rhs is None else rhs.add_(r)
                del r
        rhs *= inv_den
        return torch.fft.irfftn(rhs, s=grid, dim=dims)

    def prox(k: int, v: Tensor) -> Tensor:
        """prox of fₖ/ρ on the region; never returns `v` itself."""
        term = terms[k]
        if term.region is None:
            z = term.function.prox(v, t)
            if z is v or z.data_ptr() == v.data_ptr():
                z = z.clone()  # v is updated in place afterwards
            return z
        index = (slice(None), *term.region)
        z = v.clone()
        z[index] = term.function.prox(v[index], t)
        return z

    def sq_dist(a: Tensor, b: Tensor | None = None) -> float:
        """‖a − b‖² (‖a‖² without b), one component at a time."""
        return sum(
            float(torch.linalg.vector_norm(a[c] if b is None else a[c] - b[c]))
            ** 2
            for c in range(a.shape[0])
        )

    def objective(splits: list[Tensor]) -> float:
        total = 0.0
        for term, z in zip(terms, splits, strict=True):
            zr = z if term.region is None else z[(slice(None), *term.region)]
            total += float(term.function.value(zr))
        return total

    def dual_residual(dx: Tensor | None) -> float:
        if order == "x-first":
            return float(rho) * _norm(adjoint_sum(duals))
        spec = torch.fft.rfftn(dx, dim=dims)
        total = sum(sq_dist(apply(sym, spec, dx)) for sym in symbols)
        return float(rho) * total**0.5

    x = (
        torch.zeros(grid, dtype=dtype, device=device)
        if x0 is None
        else x0.to(dtype=dtype, device=device)
    )
    if tuple(x.shape) != grid:
        raise ValueError(f"x0 has shape {tuple(x.shape)}, expected {grid}")
    duals = [
        torch.zeros(
            (1 if sym is None else len(sym), *grid), dtype=dtype, device=device
        )
        for sym in symbols
    ]
    if order == "x-first":
        splits = [torch.zeros_like(u) for u in duals]
    else:
        spec = torch.fft.rfftn(x, dim=dims)
        splits = [
            prox(k, apply(sym, spec, x)) for k, sym in enumerate(symbols)
        ]
    if trace is not None:
        for key in ("change", "residual", "dual_residual", "objective"):
            trace[key] = []
    change = float("inf")
    primal = float("inf")
    dual = float("inf")
    n = 0
    while n < max_iter:
        x_new = solve(splits, duals)
        spec = torch.fft.rfftn(x_new, dim=dims)
        res2 = 0.0
        for k, sym in enumerate(symbols):
            v = apply(sym, spec, x_new)  # Aₖ x
            if order == "x-first":
                u = duals[k]
                splits[k] = None  # free zₖ before the prox
                v += u  # Aₖ x + uₖ
                z = prox(k, v)
                v -= z  # new uₖ = Aₖ x + uₖ − zₖ
                res2 += sq_dist(v, u)  # ‖Aₖ x − zₖ‖ = ‖Δuₖ‖
                splits[k], duals[k] = z, v
                del u, z
            else:
                v -= splits[k]  # Aₖ x − zₖ
                res2 += sq_dist(v)
                duals[k] += v
                v += splits[k]
                splits[k] = None
                v += duals[k]
                splits[k] = prox(k, v)
            del v
        del spec
        n += 1
        prev = _norm(x)
        dx = x_new - x
        step = _norm(dx)
        x = x_new
        change = step / prev if prev > 0 else float("nan")
        primal = res2**0.5
        stop = n >= max_iter or (
            tol is not None and prev > 0 and change <= tol
        )
        if trace is not None or stop:
            dual = dual_residual(dx)
        del dx
        if trace is not None:
            trace["change"].append(change)
            trace["residual"].append(primal)
            trace["dual_residual"].append(dual)
            trace["objective"].append(objective(splits))
        if callback is not None:
            callback(n, x)
        if stop:
            break
    if change != change:  # nan: never measured
        change = float("inf")
    return AdmmResult(x, splits, duals, n, change, primal, dual)
