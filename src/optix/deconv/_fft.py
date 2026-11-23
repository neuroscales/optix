"""FFT convolution of a block with a padded, extended object domain."""

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["PaddedConvolution", "next_fast_len", "transfer_function"]


def next_fast_len(n: int) -> int:
    """Smallest integer m ≥ n of the form 2ᵃ3ᵇ5ᶜ.

    Parameters
    ----------
    n : int
        Minimum length (n ≥ 1).

    Returns
    -------
    int
        A length for which FFTs are fast.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    best = 1
    while best < n:
        best *= 2
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            m = p35
            while m < n:
                m *= 2
            best = min(best, m)
            p35 *= 3
        p5 *= 5
    return best


def transfer_function(psf: Tensor, grid: Sequence[int]) -> Tensor:
    """Transfer function of a PSF on a periodic grid (half spectrum).

    The PSF S, of shape K and centred at index `K // 2`, is placed on a
    grid of shape N ≥ K with its centre at the origin (wrapped around),
    and

        H(ν) = ∑ⱼ S(j − K//2) e^{−2πi ν·j},   νₐ = kₐ / Nₐ,

    is returned for the frequencies of `torch.fft.rfftn` (the last dim
    holds N₋₁ // 2 + 1 of them). Multiplying the spectrum of an image by
    H is the circular convolution with S.

    Parameters
    ----------
    psf : Tensor
        Point spread function, `(*K)`, used as given.
    grid : sequence of int
        Grid shape N, one entry per PSF dim, with N ≥ K.

    Returns
    -------
    Tensor
        Complex tensor of shape `(*N[:-1], N[-1] // 2 + 1)`.
    """
    grid = tuple(int(n) for n in grid)
    if len(grid) != psf.ndim:
        raise ValueError(
            f"grid has {len(grid)} dims but the PSF has {psf.ndim}"
        )
    if any(n < k for n, k in zip(grid, psf.shape, strict=True)):
        raise ValueError(
            f"grid {grid} is smaller than the PSF {tuple(psf.shape)}"
        )
    buf = psf.new_zeros(grid)
    buf[tuple(slice(0, k) for k in psf.shape)] = psf
    dims = tuple(range(psf.ndim))
    buf = torch.roll(
        buf, shifts=tuple(-(k // 2) for k in psf.shape), dims=dims
    )
    return torch.fft.rfftn(buf, dim=dims)


class PaddedConvolution:
    """Blur operator H of a block, by FFT on a padded grid.

    The observed block y has shape L (the trailing `psf.ndim` dims). The
    object f lives on a larger grid of shape N ≥ L + K − 1 (K the PSF
    shape), rounded up to 2ᵃ3ᵇ5ᶜ along each dim, so that light from
    outside the block that the PSF spreads into it is modelled. The
    operator is

        (H f)ᵢ = ∑ⱼ S(i − j) fⱼ,   i in the block,

    computed as a circular convolution on the N grid and cropped to the
    block. With N ≥ L + K − 1 the wrap-around never reaches the block,
    so the circular and linear models agree. Object index j and data
    index i refer to the same location for j < L; the extra object pixels
    j ≥ L hold the margin around the block (periodically, on both sides).

    The PSF is centred at index `K // 2` along each dim.

    Attributes
    ----------
    shape : tuple of int
        Block shape L.
    grid : tuple of int
        Padded object grid N.
    norm : Tensor
        Hᵀ1, the adjoint applied to a block of ones, on the N grid. It
        equals ∑S inside the block, away from its edges, and decays to 0
        in the margin.
    """

    def __init__(
        self,
        psf: Tensor,
        shape: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Precompute the transfer function and the normaliser.

        Parameters
        ----------
        psf : Tensor
            Point spread function, `(*K)`, used as given (not normalised).
        shape : sequence of int
            Block shape L, with `len(shape) == psf.ndim`.
        dtype, device : optional
            Working backend (default: those of `psf`).
        """
        shape = tuple(int(s) for s in shape)
        if len(shape) != psf.ndim:
            raise ValueError(
                f"Block has {len(shape)} dims but the PSF has {psf.ndim}"
            )
        dtype = psf.dtype if dtype is None else dtype
        device = psf.device if device is None else device
        psf = psf.to(dtype=dtype, device=device)
        self.ndim = psf.ndim
        self.shape = shape
        self.grid = tuple(
            next_fast_len(n + k - 1)
            for n, k in zip(shape, psf.shape, strict=True)
        )
        self._dims = tuple(range(-self.ndim, 0))
        self.otf = transfer_function(psf, self.grid)
        self.norm = self.adjoint(psf.new_ones(shape))

    def _crop(self, x: Tensor) -> Tensor:
        lead = (slice(None),) * (x.ndim - self.ndim)
        return x[lead + tuple(slice(0, n) for n in self.shape)]

    def pad(self, x: Tensor) -> Tensor:
        """Zero-pad a block `(..., *L)` to the grid `(..., *N)`."""
        pad = []
        sizes = zip(self.shape, self.grid, strict=True)
        for n, m in reversed(list(sizes)):
            pad += [0, m - n]
        return torch.nn.functional.pad(x, pad)

    def crop(self, x: Tensor) -> Tensor:
        """Restrict an object `(..., *N)` to the block `(..., *L)`."""
        return self._crop(x)

    def forward(self, f: Tensor) -> Tensor:
        """H f: object `(..., *N)` to blurred block `(..., *L)`."""
        spec = torch.fft.rfftn(f, dim=self._dims) * self.otf
        return self._crop(torch.fft.irfftn(spec, s=self.grid, dim=self._dims))

    def adjoint(self, r: Tensor) -> Tensor:
        """Hᵀ r: block `(..., *L)` to object `(..., *N)` (correlation)."""
        spec = torch.fft.rfftn(self.pad(r), dim=self._dims) * self.otf.conj()
        return torch.fft.irfftn(spec, s=self.grid, dim=self._dims)
