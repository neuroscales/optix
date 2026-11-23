"""Linear operators: protocols, the identity and periodic convolutions."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "Circulant",
    "Convolution",
    "Identity",
    "LinearOperator",
    "as_circulant",
]


def _complex(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


@runtime_checkable
class LinearOperator(Protocol):
    """A linear operator A with C output components.

    It maps an image `(..., *N)` to `(..., C, *N)`, the components
    stacked along dim `-(d + 1)` with d = len(N).
    """

    @property
    def norm2(self) -> float:
        """Upper bound on ‖A‖² (the largest eigenvalue of AᵀA)."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """A x, `(..., *N)` → `(..., C, *N)`."""
        ...

    def adjoint(self, y: Tensor) -> Tensor:
        """Aᵀ y, `(..., C, *N)` → `(..., *N)`."""
        ...


@runtime_checkable
class Circulant(LinearOperator, Protocol):
    """A [`LinearOperator`][optix.optim.LinearOperator] diagonal in Fourier.

    Each component is a periodic convolution on the grid N, given by
    its transfer function on the half spectrum of `torch.fft.rfftn`.
    """

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        """Transfer function of each of the C components.

        Complex tensors of the precision of `dtype` that broadcast to the
        half spectrum `(*N[:-1], N[-1] // 2 + 1)`.
        """
        ...

    def gram(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> Tensor:
        """∑_c |symbol_c|², the transfer function of AᵀA (real)."""
        ...


class _CirculantBase:
    """Shared `gram` and component-axis helpers."""

    ndim: int

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        raise NotImplementedError

    def gram(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> Tensor:
        """∑_c |symbol_c|², the transfer function of AᵀA (real)."""
        total = None
        for s in self.symbols(grid, dtype=dtype, device=device):
            p = s.real.square() + s.imag.square()
            total = p if total is None else total + p
        return total


class Identity(_CirculantBase):
    """The identity on images of d dims, as a one-component operator."""

    def __init__(self, ndim: int) -> None:
        """Set the number of image dims.

        Parameters
        ----------
        ndim : int
            Number d of trailing image dims.
        """
        self.ndim = int(ndim)
        self.components = 1

    @property
    def norm2(self) -> float:
        """‖I‖² = 1."""
        return 1.0

    def forward(self, x: Tensor) -> Tensor:
        """x, `(..., *N)` → `(..., 1, *N)` (a new tensor)."""
        return x.unsqueeze(-self.ndim - 1).clone()

    def adjoint(self, y: Tensor) -> Tensor:
        """y, `(..., 1, *N)` → `(..., *N)`."""
        return y.select(-self.ndim - 1, 0).clone()

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        """A single transfer function equal to 1."""
        shape = (1,) * len(tuple(grid))
        return [torch.ones(shape, dtype=_complex(dtype), device=device)]


class Convolution(_CirculantBase):
    """Periodic convolution on a grid N, given by its transfer function.

    Wraps a transfer function on the half spectrum of `torch.fft.rfftn`
    (for example the `otf` of
    [`PaddedConvolution`][optix.deconv.utils.PaddedConvolution]) as a
    one-component [`Circulant`][optix.optim.Circulant] operator.
    """

    def __init__(self, transfer: Tensor, grid: Sequence[int]) -> None:
        """Set the transfer function.

        Parameters
        ----------
        transfer : Tensor
            Transfer function Â, broadcastable to the half spectrum
            `(*N[:-1], N[-1] // 2 + 1)`; real tensors are converted to
            complex.
        grid : sequence of int
            Periodic grid N.
        """
        self.grid = tuple(int(n) for n in grid)
        self.ndim = len(self.grid)
        if not transfer.is_complex():
            transfer = transfer.to(_complex(transfer.dtype))
        half = (*self.grid[:-1], self.grid[-1] // 2 + 1)
        try:
            ok = torch.broadcast_shapes(transfer.shape, half) == half
        except RuntimeError:
            ok = False
        if not ok:
            raise ValueError(
                f"transfer function of shape {tuple(transfer.shape)} does "
                f"not broadcast to the half spectrum {half}"
            )
        self.transfer = transfer
        self.components = 1
        self._dims = tuple(range(-self.ndim, 0))

    @property
    def norm2(self) -> float:
        """‖A‖² = max |Â|²."""
        t = self.transfer
        return float((t.real.square() + t.imag.square()).max())

    def forward(self, x: Tensor) -> Tensor:
        """A x, `(..., *N)` → `(..., 1, *N)`."""
        spec = torch.fft.rfftn(x, dim=self._dims) * self.transfer
        out = torch.fft.irfftn(spec, s=self.grid, dim=self._dims)
        return out.unsqueeze(-self.ndim - 1)

    def adjoint(self, y: Tensor) -> Tensor:
        """Aᵀ y, `(..., 1, *N)` → `(..., *N)`."""
        y = y.select(-self.ndim - 1, 0)
        spec = torch.fft.rfftn(y, dim=self._dims) * self.transfer.conj()
        return torch.fft.irfftn(spec, s=self.grid, dim=self._dims)

    def symbols(
        self,
        grid: Sequence[int],
        *,
        dtype: torch.dtype = torch.float64,
        device: object = None,
    ) -> list[Tensor]:
        """The transfer function, cast to the complex dtype of `dtype`."""
        if tuple(int(n) for n in grid) != self.grid:
            raise ValueError(
                f"Convolution is defined on the grid {self.grid}, "
                f"not {tuple(grid)}"
            )
        return [self.transfer.to(dtype=_complex(dtype), device=device)]


def as_circulant(operator: object, grid: Sequence[int]) -> Circulant:
    """The [`Circulant`][optix.optim.Circulant] operator meant by `operator`.

    Parameters
    ----------
    operator : None, Tensor or Circulant
        None: the [`Identity`][optix.optim.Identity]; a tensor: the
        transfer function of a [`Convolution`][optix.optim.Convolution];
        anything else must implement `Circulant`.
    grid : sequence of int
        Periodic grid N.

    Returns
    -------
    Circulant
    """
    grid = tuple(int(n) for n in grid)
    if operator is None:
        return Identity(len(grid))
    if isinstance(operator, Tensor):
        return Convolution(operator, grid)
    if not isinstance(operator, Circulant):
        raise TypeError(
            "operator must be None, a transfer function or a Circulant "
            "operator (forward, adjoint, norm2, symbols, gram): got "
            f"{type(operator).__name__}"
        )
    return operator
