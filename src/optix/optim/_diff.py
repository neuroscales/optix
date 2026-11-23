"""Finite differences: first and second derivatives, and their symbols."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "Gradient",
    "Hessian",
    "diff",
    "diff_adjoint",
    "rfft_frequencies",
]

_BOUNDARY = ("periodic", "neumann")
#: Second-difference stencils of `Hessian`.
_STENCIL = ("forward",)


def _check_boundary(boundary: str) -> str:
    if boundary not in _BOUNDARY:
        raise ValueError(f"boundary must be one of {_BOUNDARY}: {boundary!r}")
    return boundary


def diff(x: Tensor, dim: int, *, boundary: str = "periodic") -> Tensor:
    """Forward difference along `dim`: (D x)ᵢ = xᵢ₊₁ − xᵢ.

    Parameters
    ----------
    x : Tensor
        Image.
    dim : int
        Dim along which to differentiate.
    boundary : {"periodic", "neumann"}
        `"periodic"`: xₙ = x₀ (wrap-around). `"neumann"`: the difference
        is 0 at the last index (the image is extended by replication).

    Returns
    -------
    Tensor
        D x, shaped like `x`.
    """
    if _check_boundary(boundary) == "periodic":
        return torch.roll(x, -1, dim) - x
    out = torch.zeros_like(x)
    n = x.shape[dim]
    out.narrow(dim, 0, n - 1).copy_(
        x.narrow(dim, 1, n - 1) - x.narrow(dim, 0, n - 1)
    )
    return out


def diff_adjoint(y: Tensor, dim: int, *, boundary: str = "periodic") -> Tensor:
    """Adjoint of [`diff`][optix.optim.diff]: (Dᵀ y)ᵢ = yᵢ₋₁ − yᵢ.

    Parameters
    ----------
    y : Tensor
        Differences.
    dim : int
        Dim of the difference.
    boundary : {"periodic", "neumann"}
        As in [`diff`][optix.optim.diff] (with `"neumann"`, y₋₁ = 0 and
        the last entry of y is ignored).

    Returns
    -------
    Tensor
        Dᵀ y, shaped like `y`.
    """
    if _check_boundary(boundary) == "periodic":
        return torch.roll(y, 1, dim) - y
    n = y.shape[dim]
    out = torch.zeros_like(y)
    out.narrow(dim, 1, n - 1).copy_(y.narrow(dim, 0, n - 1))
    out.narrow(dim, 0, n - 1).sub_(y.narrow(dim, 0, n - 1))
    return out


def rfft_frequencies(
    grid: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> list[Tensor]:
    """Frequencies νₐ = kₐ/Nₐ of `torch.fft.rfftn` on a grid, per axis.

    Parameters
    ----------
    grid : sequence of int
        Grid shape N.
    dtype, device : optional
        Backend of the result.

    Returns
    -------
    list of Tensor
        One tensor per axis, shaped to broadcast against the half
        spectrum `(*N[:-1], N[-1] // 2 + 1)`.
    """
    grid = tuple(int(n) for n in grid)
    out = []
    for a, n in enumerate(grid):
        if a == len(grid) - 1:
            f = torch.fft.rfftfreq(n, dtype=dtype, device=device)
        else:
            f = torch.fft.fftfreq(n, dtype=dtype, device=device)
        shape = [1] * len(grid)
        shape[a] = f.numel()
        out.append(f.reshape(shape))
    return out


def _complex(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


class _Differences:
    """Shared parts of the difference operators."""

    def __init__(
        self,
        ndim: int,
        *,
        weights: Sequence[float] | None = None,
        boundary: str = "periodic",
    ) -> None:
        self.ndim = int(ndim)
        if self.ndim < 1:
            raise ValueError("ndim must be positive")
        if weights is None:
            weights = (1.0,) * self.ndim
        self.weights = tuple(float(w) for w in weights)
        if len(self.weights) != self.ndim:
            raise ValueError("weights must have one entry per dim")
        if any(not w >= 0 for w in self.weights):
            raise ValueError("weights must be non-negative")
        self.boundary = _check_boundary(boundary)
        self._dims = tuple(range(-self.ndim, 0))

    def _d(self, x: Tensor, a: int) -> Tensor:
        return diff(x, self._dims[a], boundary=self.boundary)

    def _dt(self, y: Tensor, a: int) -> Tensor:
        return diff_adjoint(y, self._dims[a], boundary=self.boundary)

    def _factors(
        self, grid: Sequence[int], dtype: torch.dtype, device: object
    ) -> list[Tensor]:
        """wₐ D̂ₐ(ν) = wₐ (e^{2πiνₐ} − 1), broadcastable, per axis."""
        if self.boundary != "periodic":
            raise ValueError("Fourier symbols need boundary='periodic'")
        if len(grid) != self.ndim:
            raise ValueError(f"grid must have {self.ndim} dims")
        freqs = rfft_frequencies(grid, dtype=dtype, device=device)
        out = []
        for w, f in zip(self.weights, freqs, strict=True):
            phase = torch.polar(torch.ones_like(f), 2 * math.pi * f)
            out.append(w * (phase - 1).to(_complex(dtype)))
        return out

    def gram(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> Tensor:
        """∑_c |symbol_c|², the symbol of AᵀA (half spectrum, real)."""
        total = None
        for s in self.symbols(grid, dtype=dtype, device=device):
            p = s.real.square() + s.imag.square()
            total = p if total is None else total + p
        return total

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        raise NotImplementedError


class Gradient(_Differences):
    """Discrete gradient by forward differences, ∇x = (w₁D₁x, …, w_dD_dx).

    The C = d components are stacked along dim `-(d + 1)`, so an image
    `(..., *N)` maps to `(..., d, *N)`. ∑ⱼ ‖(∇x)ⱼ‖₂ is the isotropic
    total variation.
    """

    def __init__(
        self,
        ndim: int,
        *,
        weights: Sequence[float] | None = None,
        boundary: str = "periodic",
    ) -> None:
        """Configure the operator.

        Parameters
        ----------
        ndim : int
            Number d of trailing dims differentiated.
        weights : sequence of float, optional
            Weight wₐ ≥ 0 of each axis (Dₐ is replaced by wₐDₐ).
            Default: 1.
        boundary : {"periodic", "neumann"}
            See [`diff`][optix.optim.diff].
        """
        super().__init__(ndim, weights=weights, boundary=boundary)
        self.components = self.ndim

    @property
    def norm2(self) -> float:
        """Upper bound on ‖∇‖²: 4∑ₐwₐ² (attained for periodic, even N)."""
        return 4 * sum(w * w for w in self.weights)

    def forward(self, x: Tensor) -> Tensor:
        """∇x, `(..., *N)` → `(..., d, *N)`."""
        return torch.stack(
            [w * self._d(x, a) for a, w in enumerate(self.weights)],
            dim=-self.ndim - 1,
        )

    def adjoint(self, y: Tensor) -> Tensor:
        """∇ᵀy, `(..., d, *N)` → `(..., *N)`."""
        out = None
        for a, w in enumerate(self.weights):
            t = w * self._dt(y.select(-self.ndim - 1, a), a)
            out = t if out is None else out + t
        return out

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        """Transfer function of each component, wₐ(e^{2πiνₐ} − 1).

        Parameters
        ----------
        grid : sequence of int
            Periodic grid N.
        dtype, device : optional
            Real dtype (the result is complex of the same precision) and
            device.

        Returns
        -------
        list of Tensor
            d complex tensors that broadcast to the half spectrum of
            `torch.fft.rfftn` on N.
        """
        return self._factors(grid, dtype, device)


class Hessian(_Differences):
    """Discrete Hessian by forward differences, packed as a vector.

    The second differences are compositions of forward differences,
    D_ab = D_a D_b (so D_aa xᵢ = xᵢ₊₂ − 2xᵢ₊₁ + xᵢ), as in Ikoma et al.
    (2018, Eq. S5). The components are packed as in their Eq. (S7): the
    d diagonal terms D_aa, then the off-diagonal terms √2·D_ab for
    a < b, C = d(d + 1)/2 in all, stacked along dim `-(d + 1)`. Then
    ‖(Hx)ⱼ‖₂ is the Frobenius norm of the d × d Hessian matrix at voxel
    j. With axis weights wₐ, Dₐ is replaced by wₐDₐ.
    """

    def __init__(
        self,
        ndim: int,
        *,
        weights: Sequence[float] | None = None,
        boundary: str = "periodic",
        stencil: str = "forward",
    ) -> None:
        """Configure the operator.

        Parameters
        ----------
        ndim : int
            Number d of trailing dims differentiated.
        weights : sequence of float, optional
            Weight wₐ ≥ 0 of each axis. Default: 1.
        boundary : {"periodic", "neumann"}
            See [`diff`][optix.optim.diff].
        stencil : {"forward"}
            Second-difference stencil. `"forward"`: D_ab = D_a D_b, the
            composition of two forward differences (Ikoma et al. 2018,
            Eq. S5).
        """
        super().__init__(ndim, weights=weights, boundary=boundary)
        if stencil not in _STENCIL:
            raise ValueError(f"stencil must be one of {_STENCIL}: {stencil!r}")
        self.stencil = stencil
        d = self.ndim
        self.pairs = [(a, a) for a in range(d)] + [
            (a, b) for a in range(d) for b in range(a + 1, d)
        ]
        self.components = len(self.pairs)

    @property
    def norm2(self) -> float:
        """Upper bound on ‖H‖²: (4∑ₐwₐ²)²."""
        return (4 * sum(w * w for w in self.weights)) ** 2

    def _scale(self, a: int, b: int) -> float:
        w = self.weights[a] * self.weights[b]
        return w if a == b else math.sqrt(2) * w

    def forward(self, x: Tensor) -> Tensor:
        """H x, `(..., *N)` → `(..., C, *N)`."""
        first = [self._d(x, a) for a in range(self.ndim)]
        comps = [
            self._scale(a, b) * self._d(first[b], a) for a, b in self.pairs
        ]
        return torch.stack(comps, dim=-self.ndim - 1)

    def adjoint(self, y: Tensor) -> Tensor:
        """Hᵀ y, `(..., C, *N)` → `(..., *N)`."""
        out = None
        for c, (a, b) in enumerate(self.pairs):
            yc = self._scale(a, b) * y.select(-self.ndim - 1, c)
            t = self._dt(self._dt(yc, a), b)
            out = t if out is None else out + t
        return out

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        """Transfer function of each component (broadcastable).

        Parameters
        ----------
        grid : sequence of int
            Periodic grid N.
        dtype, device : optional
            Real dtype (the result is complex of the same precision) and
            device.

        Returns
        -------
        list of Tensor
            C complex tensors, D̂_aD̂_b (×√2 off the diagonal), that
            broadcast to the half spectrum of `torch.fft.rfftn` on N.
        """
        unit = _Differences(self.ndim, boundary=self.boundary)
        f = unit._factors(grid, dtype, device)
        return [self._scale(a, b) * (f[a] * f[b]) for a, b in self.pairs]
