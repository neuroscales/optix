"""Generalised Anscombe transform and its exact unbiased inverse.

Implemented from the papers of Mäkitalo & Foi (2011, 2011 letter, 2013);
the authors' Matlab code, under a non-commercial licence, was not used.
"""

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from .._estimator import Tags
from ..estimates import SensorEstimate
from ..io import ArrayLike, WritableArrayLike
from ._base import CalibrationEstimator

__all__ = [
    "GeneralizedAnscombe",
    "anscombe",
    "closed_form_inverse",
    "exact_inverse_table",
    "inverse_anscombe",
]


def anscombe(
    z: Tensor,
    alpha: Tensor | float = 1.0,
    sigma: Tensor | float = 0.0,
    mu: Tensor | float = 0.0,
) -> Tensor:
    """Generalised Anscombe transform (pointwise).

    For z = α·Poisson(y/α) + 𝒩(μ, σ²) (Mäkitalo & Foi 2013, Eq. (3)):

        f(z) = (2/α)·√max(0, α·z + 3/8·α² + σ² − α·μ),

    whose noise is approximately standard normal.

    Parameters
    ----------
    z : Tensor
        Data.
    alpha, sigma, mu : Tensor or float
        Gain α, Gaussian standard deviation σ and mean μ; scalars or maps
        broadcastable to `z`.

    Returns
    -------
    Tensor
    """
    arg = alpha * z + 0.375 * alpha * alpha + sigma * sigma - alpha * mu
    return 2 / alpha * arg.clamp_min(0).sqrt()


def closed_form_inverse(d: Tensor, s2: Tensor | float = 0.0) -> Tensor:
    """Closed-form approximation of the exact unbiased inverse.

    Mäkitalo & Foi 2011 letter, Eq. (6), generalised in Mäkitalo & Foi
    2013, Eq. (21) (normalised units, σ = σ̂/α):

        Ĩ_σ(D) = Ĩ_0(D) − σ²,
        Ĩ_0(D) = D²/4 + (1/4)·√(3/2)·D⁻¹ − (11/8)·D⁻² + (5/8)·√(3/2)·D⁻³ − 1/8,

    with Ĩ_0(D) = 0 for D < 2·√(3/8) (letter, Section II). As published,
    Ĩ_σ is not clipped further: it equals −σ² below 2·√(3/8).

    Parameters
    ----------
    d : Tensor
        Transformed data D.
    s2 : Tensor or float
        Normalised Gaussian variance σ² = σ̂²/α².

    Returns
    -------
    Tensor
    """
    c = math.sqrt(1.5)
    dd = d.clamp_min(2 * math.sqrt(0.375))
    inv = dd.reciprocal()
    i0 = (
        0.25 * dd * dd
        + inv * (0.25 * c + inv * (-11 / 8 + inv * (5 / 8 * c)))
        - 0.125
    )
    i0 = torch.where(d < 2 * math.sqrt(0.375), torch.zeros_like(i0), i0)
    return i0 - s2


def _asymptotic(d: Tensor, s2: Tensor | float) -> Tensor:
    """Asymptotically unbiased inverse D²/4 − 1/8 − σ² (2013, Eq. (22))."""
    return 0.25 * d * d - 0.125 - s2


def exact_inverse_table(
    s2: Tensor,
    *,
    ymax: float = 200.0,
    ny: int = 2000,
    nd: int = 4096,
    nquad: int = 2001,
) -> tuple[Tensor, Tensor]:
    """Lookup table of the exact unbiased inverse, normalised units.

    For z = Poisson(y) + 𝒩(0, σ²) and f_σ(z) = 2·√max(0, z + 3/8 + σ²),
    computes (Mäkitalo & Foi 2013, Eq. (10))

        E{f_σ(z) | y, σ} = ∑ₖ yᵏ·e⁻ʸ/k! · ∫ f_σ(k + σ·n)·φ(n) dn

    on a grid of y (trapezoidal rule over n ∈ [−10, 10]) and inverts it
    on a uniform grid of D.

    Parameters
    ----------
    s2 : (S,) Tensor
        Grid of σ² (normalised: σ̂²/α²).
    ymax : float
        Largest y of the table; above, [`inverse_anscombe`][] uses the
        closed form Ĩ_0(D) − σ².
    ny, nd, nquad : int
        Sizes of the y grid, of the D grid and of the quadrature.

    Returns
    -------
    dgrid : (nd,) Tensor
        Uniform grid of D from 0 to the smallest E{f | ymax}.
    table : (S, nd) Tensor
        y such that E{f_σ(z) | y} = D (0 below E{f | 0}).

    !!! note
        Computed in float64 on the CPU.
    """
    f64 = {"dtype": torch.float64, "device": "cpu"}
    s2 = torch.as_tensor(s2).to(**f64).reshape(-1)
    y = torch.linspace(0, math.sqrt(ymax), ny, **f64).square()
    kmax = int(math.ceil(ymax + 12 * math.sqrt(ymax) + 30))
    k = torch.arange(kmax + 1, **f64)
    # Poisson pmf (ny, K)
    logy = torch.where(y > 0, y.log(), -math.inf)
    logp = k * logy[:, None] - y[:, None] - torch.lgamma(k + 1)
    logp[0, 0] = 0.0
    pmf = torch.where(torch.isnan(logp), 0.0, logp.exp())
    n = torch.linspace(-10, 10, nquad, **f64)
    phi = torch.exp(-0.5 * n * n)
    phi = phi / torch.trapezoid(phi, n)
    expect = []
    for s in s2.tolist():
        if s > 0:
            t = k[:, None] + math.sqrt(s) * n
            g = torch.trapezoid(
                2 * (t + 0.375 + s).clamp_min(0).sqrt() * phi, n, dim=-1
            )
        else:
            g = 2 * (k + 0.375).sqrt()
        expect.append(pmf @ g)
    expect = torch.stack(expect)  # (S, ny), increasing in y
    dmax = float(expect[:, -1].min())
    dgrid = torch.linspace(0, dmax, nd, **f64)
    table = torch.stack(
        [
            torch.from_numpy(
                np.interp(dgrid.numpy(), e.numpy(), y.numpy(), left=0.0)
            )
            for e in expect
        ]
    )
    return dgrid, table


def _lookup(
    d: Tensor, s2: Tensor, sgrid: Tensor, dgrid: Tensor, table: Tensor
) -> Tensor:
    """Bilinear interpolation of `table` at `(s2, d)` (uniform grids)."""
    nd = dgrid.numel()
    dmax = dgrid[-1]
    u = (d / dmax * (nd - 1)).clamp(0, nd - 1)
    i0 = u.floor().clamp(max=nd - 2).long()
    fu = u - i0
    ns = sgrid.numel()
    if ns == 1:
        row = table[0]
        return row[i0] * (1 - fu) + row[i0 + 1] * fu
    v = ((s2 - sgrid[0]) / (sgrid[-1] - sgrid[0]) * (ns - 1)).clamp(0, ns - 1)
    j0 = v.floor().clamp(max=ns - 2).long()
    fv = v - j0
    j0, i0 = torch.broadcast_tensors(j0, i0)
    fv, fu = torch.broadcast_tensors(fv, fu)
    flat = table.reshape(-1)
    base = j0 * nd + i0

    def at(off: int) -> Tensor:
        return flat[base + off]

    lo = at(0) * (1 - fu) + at(1) * fu
    hi = at(nd) * (1 - fu) + at(nd + 1) * fu
    return lo * (1 - fv) + hi * fv


def inverse_anscombe(
    d: Tensor,
    alpha: Tensor | float = 1.0,
    sigma: Tensor | float = 0.0,
    mu: Tensor | float = 0.0,
    *,
    method: str = "exact",
    table: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """Inverse of the generalised Anscombe transform.

    Works in normalised units σ = σ̂/α (Mäkitalo & Foi 2013, Eq. (4)) and
    returns α·ŷ + μ, an estimate of E[z].

    Parameters
    ----------
    d : Tensor
        Transformed (typically denoised) data D.
    alpha, sigma, mu : Tensor or float
        As in [`anscombe`][].
    method : {"exact", "closed-form", "asymptotic", "algebraic"}
        The inverses of Mäkitalo & Foi 2013:

        - `"exact"`: exact unbiased inverse I_σ (Eqs. (9)-(10)),
          interpolated in a table of E{f_σ(z) | y, σ} and 0 below
          E{f_σ | 0, σ} (the ML inverse, Eq. (14)); outside the table
          (y > ymax), the paper's asymptotic form I_0(D) − σ² (Section
          III-E, Eq. (15)) with the closed form of I_0;
        - `"closed-form"`: [`closed_form_inverse`][], Eq. (21);
        - `"asymptotic"`: D²/4 − 1/8 − σ², Eq. (22);
        - `"algebraic"`: D²/4 − 3/8 − σ² (biased).
    table : tuple, optional
        `(sgrid, dgrid, table)` from [`exact_inverse_table`][] (with
        `sgrid` the uniform grid of normalised σ²), to reuse; default:
        built for the range of σ/α.

    Returns
    -------
    Tensor
    """
    alpha = torch.as_tensor(alpha, dtype=d.dtype, device=d.device)
    sigma = torch.as_tensor(sigma, dtype=d.dtype, device=d.device)
    s2 = (sigma / alpha).square()
    if method == "algebraic":
        y = 0.25 * d * d - 0.375 - s2
    elif method == "closed-form":
        y = closed_form_inverse(d, s2)
    elif method == "asymptotic":
        y = _asymptotic(d, s2)
    elif method == "exact":
        if table is None:
            table = build_table(s2)
        sgrid, dgrid, tab = (t.to(d) for t in table)
        y = torch.where(
            d > dgrid[-1],
            closed_form_inverse(d, s2),
            _lookup(d, s2, sgrid, dgrid, tab),
        )
    else:
        raise ValueError(f"Unknown method {method!r}")
    return alpha * y + mu


def build_table(s2: Tensor, n: int = 33) -> tuple[Tensor, Tensor, Tensor]:
    """Exact-inverse table for normalised variances in the range of `s2`.

    Returns `(sgrid, dgrid, table)`: `sgrid` is a uniform grid of σ² with
    `n` points (one when `s2` is constant).
    """
    lo, hi = float(s2.min()), float(s2.max())
    if hi - lo <= 1e-9 * max(1.0, hi):
        sgrid = torch.tensor([lo], dtype=torch.float64)
    else:
        sgrid = torch.linspace(lo, hi, n, dtype=torch.float64)
    dgrid, tab = exact_inverse_table(sgrid)
    return sgrid, dgrid, tab


class GeneralizedAnscombe(CalibrationEstimator):
    """Variance-stabilising transform for Poisson-Gaussian data.

    Model (Mäkitalo & Foi 2013, Eq. (1)): z = α·Poisson(y/α) + 𝒩(μ, σ²),
    i.e. a camera with gain α (ADU per electron), offset μ and read
    variance σ² (a [`SensorEstimate`][optix.estimates.SensorEstimate]).
    [`apply`][optix.sensor.GeneralizedAnscombe.apply] computes
    [`anscombe`][], [`inverse`][optix.sensor.GeneralizedAnscombe.inverse]
    its inverse.

    !!! note
        `fit(None, init=sensor)` only reads the parameters (no data
        needed). The exact-inverse table is computed once (float64, CPU)
        and interpolated on the data device.

    !!! references "References"
        1. Mäkitalo M, Foi A. *Optimal inversion of the Anscombe
           transformation in low-count Poisson image denoising.* IEEE Trans
           Image Process **20**(1):99–109, 2011.
           [doi:10.1109/TIP.2010.2056693](https://doi.org/10.1109/TIP.2010.2056693)
        2. Mäkitalo M, Foi A. *A closed-form approximation of the exact
           unbiased inverse of the Anscombe variance-stabilizing
           transformation.* IEEE Trans Image Process **20**(9):2697–2698,
           2011.
           [doi:10.1109/TIP.2011.2121085](https://doi.org/10.1109/TIP.2011.2121085)
        3. Mäkitalo M, Foi A. *Optimal inversion of the generalized
           Anscombe transformation for Poisson-Gaussian noise.* IEEE Trans
           Image Process **22**(1):91–103, 2013.
           [doi:10.1109/TIP.2012.2202675](https://doi.org/10.1109/TIP.2012.2202675)

        ??? quote "BibTeX"
            ```bibtex
            @article{makitalo2011optimal,
              author  = {Mäkitalo, Markku and Foi, Alessandro},
              title   = {Optimal inversion of the {Anscombe} transformation
                         in low-count {Poisson} image denoising},
              journal = {IEEE Transactions on Image Processing},
              volume  = {20}, number = {1}, pages = {99--109}, year = {2011},
              doi     = {10.1109/TIP.2010.2056693}
            }
            @article{makitalo2011closed,
              author  = {Mäkitalo, Markku and Foi, Alessandro},
              title   = {A closed-form approximation of the exact unbiased
                         inverse of the {Anscombe} variance-stabilizing
                         transformation},
              journal = {IEEE Transactions on Image Processing},
              volume  = {20}, number = {9}, pages = {2697--2698},
              year    = {2011}, doi = {10.1109/TIP.2011.2121085}
            }
            @article{makitalo2013optimal,
              author  = {Mäkitalo, Markku and Foi, Alessandro},
              title   = {Optimal inversion of the generalized {Anscombe}
                         transformation for {Poisson-Gaussian} noise},
              journal = {IEEE Transactions on Image Processing},
              volume  = {22}, number = {1}, pages = {91--103}, year = {2013},
              doi     = {10.1109/TIP.2012.2202675}
            }
            ```
    """

    tags = Tags(
        targets=("vst",),
        streaming="pointwise",
        noise="poisson-gaussian",
        reference="doi:10.1109/TIP.2010.2056693; doi:10.1109/TIP.2012.2202675",
    )
    paper = {
        "alpha": None,
        "sigma": None,
        "mu": None,
        "inverse": "exact",
        "per_pixel": False,
    }
    recommended = dict(paper)

    def __init__(
        self,
        *,
        alpha: float | None = None,
        sigma: float | None = None,
        mu: float | None = None,
        inverse: str | None = None,
        per_pixel: bool | None = None,
        faithful: bool = True,
        dtype: object = None,
        device: object = None,
        layout: object = None,
    ) -> None:
        """Configure the transform.

        Parameters
        ----------
        alpha, sigma, mu : float, optional
            Noise parameters; alternatively `fit(init=sensor)`.
        inverse : {"exact", "closed-form", "asymptotic", "algebraic"}
            Inverse used by `inverse` (see [`inverse_anscombe`][]); all
            four are defined in Mäkitalo & Foi 2013. Paper: `"exact"`.
        per_pixel : bool
            Extension: use the per-pixel maps of the sensor estimate.
            Paper: False (scalar parameters: the means of the maps).
        faithful, dtype, device, layout
            See [`Estimator`][optix.Estimator].
        """
        super().__init__(dtype=dtype, device=device, layout=layout)
        self._resolve(faithful, locals())
        self._table: tuple[Tensor, Tensor, Tensor] | None = None
        o = self.options
        if o["alpha"] is not None:
            self.estimates = {
                "sensor": SensorEstimate(
                    offset=float(o["mu"] or 0.0),
                    variance=float(o["sigma"] or 0.0) ** 2,
                    gain=float(o["alpha"]),
                )
            }

    def fit(
        self, data: ArrayLike | None = None, **kwargs: object
    ) -> "GeneralizedAnscombe":
        """Read the noise parameters from `init` (a SensorEstimate).

        Parameters
        ----------
        data : array-like, optional
            Unused.
        **kwargs
            `init` (required), `layout`, ...

        Returns
        -------
        GeneralizedAnscombe
            self
        """
        return super().fit(data, **kwargs)

    def _fit(
        self,
        data: ArrayLike,
        *,
        init: object,
        callback: Callable | None,
        verbose: bool,
    ) -> None:
        sensor = self._pick(init, SensorEstimate)
        if sensor is None:
            raise ValueError(
                "GeneralizedAnscombe.fit needs init=SensorEstimate"
            )
        for name in ("offset", "variance", "gain"):
            if getattr(sensor, name) is None:
                raise ValueError(f"init SensorEstimate lacks {name}")
        if not self.options["per_pixel"]:
            sensor = SensorEstimate(
                offset=sensor.offset.mean(),
                variance=sensor.variance.mean(),
                gain=sensor.gain.mean(),
                meta=dict(sensor.meta),
            )
        self.estimates = {"sensor": sensor}
        self._table = None

    def _params(self) -> tuple[Tensor, Tensor, Tensor]:
        if "sensor" not in self.estimates:
            raise ValueError("Not fitted: give alpha= or fit(init=sensor)")
        s = self.estimates["sensor"]
        return s.gain, s.variance.clamp_min(0).sqrt(), s.offset

    def _apply(
        self, data: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        return self._apply_maps(data, out, anscombe, *self._params())

    def inverse(
        self, data: ArrayLike, out: WritableArrayLike | None = None
    ) -> WritableArrayLike:
        """Inverse transform (method from the `inverse` option).

        Parameters
        ----------
        data : array-like
            Transformed data, in the layout of the fit.
        out : writable array-like, optional

        Returns
        -------
        writable array-like
        """
        alpha, sigma, mu = self._params()
        method = self.options["inverse"]
        if method == "exact" and self._table is None:
            s2 = (sigma.double() / alpha.double()).square()
            self._table = build_table(s2.cpu())
        table = self._table

        def fn(d: Tensor, a: Tensor, s: Tensor, m: Tensor) -> Tensor:
            return inverse_anscombe(d, a, s, m, method=method, table=table)

        if not hasattr(data, "layout"):
            data = self._labeled(data, self.layout)
        with torch.no_grad():
            return self._apply_maps(data, out, fn, alpha, sigma, mu)
