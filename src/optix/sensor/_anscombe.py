"""Generalised Anscombe transform and its exact unbiased inverse.

Mäkitalo & Foi, IEEE TIP 2011 (doi:10.1109/TIP.2010.2056693) and IEEE
TIP 2013 (doi:10.1109/TIP.2012.2202675).
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
    "exact_inverse_table",
    "inverse_anscombe",
]


def anscombe(
    z: Tensor,
    alpha: Tensor | float = 1.0,
    sigma: Tensor | float = 0.0,
    mu: Tensor | float = 0.0,
) -> Tensor:
    r"""Generalised Anscombe transform (pointwise).

    For :math:`z = \alpha\,\mathrm{Poisson}(y/\alpha) + \mathcal N(\mu,
    \sigma^2)`:

    .. math::

        f(z) = \frac{2}{\alpha}\sqrt{\max\left(0,\ \alpha z +
               \tfrac38 \alpha^2 + \sigma^2 - \alpha\mu\right)},

    whose noise is approximately standard normal.

    Parameters
    ----------
    z : Tensor
        Data.
    alpha, sigma, mu : Tensor or float
        Gain, Gaussian standard deviation and mean; scalars or maps
        broadcastable to ``z``.

    Returns
    -------
    Tensor
    """
    arg = alpha * z + 0.375 * alpha * alpha + sigma * sigma - alpha * mu
    return 2 / alpha * arg.clamp_min(0).sqrt()


def _asymptotic(d: Tensor, s: Tensor | float) -> Tensor:
    r"""Asymptotically unbiased inverse (normalised units), clipped at 0.

    A second-order expansion of :math:`E[\sqrt{z + c}]` for
    :math:`z` of mean :math:`y` and variance :math:`y + \sigma^2`,
    :math:`c = 3/8 + \sigma^2`, gives
    :math:`E[f(z) | y] = 2\sqrt{y + c} - 1 / (4\sqrt{y + c}) + O(y^{-3/2})`,
    hence :math:`y = D^2/4 - 1/8 - \sigma^2 + O(D^{-2})`.
    """
    return (0.25 * d * d - 0.125 - s).clamp_min(0)


def exact_inverse_table(
    s2: Tensor,
    *,
    ymax: float = 200.0,
    ny: int = 2000,
    nd: int = 4096,
    nquad: int = 2001,
) -> tuple[Tensor, Tensor]:
    r"""Lookup table of the exact unbiased inverse, normalised units.

    For :math:`z = \mathrm{Poisson}(y) + \mathcal N(0, \sigma^2)` and
    :math:`f(z) = 2\sqrt{\max(0, z + 3/8 + \sigma^2)}`, computes

    .. math::

        E[f(z) \mid y] = \sum_k \frac{y^k e^{-y}}{k!}
            \int f(k + \sigma n)\,\phi(n)\,dn

    on a grid of ``y`` (trapezoidal rule over ``n`` in [-10, 10]) and
    inverts it on a uniform grid of ``D``.

    Parameters
    ----------
    s2 : (S,) Tensor
        Grid of :math:`\sigma^2` (normalised: :math:`\sigma^2/\alpha^2`).
    ymax : float
        Largest ``y`` of the table; above, the asymptotic inverse is used.
    ny, nd, nquad : int
        Sizes of the ``y`` grid, of the ``D`` grid and of the quadrature.

    Returns
    -------
    dgrid : (nd,) Tensor
        Uniform grid of ``D`` from 0 to the smallest ``E[f | ymax]``.
    table : (S, nd) Tensor
        ``y`` such that ``E[f(z) | y] = D`` (0 below ``E[f | 0]``).

    Notes
    -----
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
    """Bilinear interpolation of ``table`` at ``(s2, d)`` (uniform grids)."""
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
    r"""Inverse of the generalised Anscombe transform.

    Works in normalised units :math:`\tilde\sigma = \sigma / \alpha` and
    returns :math:`\alpha\,\hat y + \mu`, an estimate of :math:`E[z]`.

    Parameters
    ----------
    d : Tensor
        Transformed (typically denoised) data.
    alpha, sigma, mu : Tensor or float
        As in :func:`anscombe`.
    method : {"exact", "asymptotic", "algebraic"}
        ``"exact"``: exact unbiased inverse :math:`E[f(z)|y] \mapsto y`
        from a lookup table (Mäkitalo & Foi 2013), with the asymptotic
        inverse above the table. ``"asymptotic"``: the closed-form
        asymptotically unbiased inverse ``D²/4 - 1/8 - σ²`` (not the
        paper's fitted closed form). ``"algebraic"``: the direct inverse
        of :func:`anscombe` (biased, but an exact round trip).
    table : tuple, optional
        ``(sgrid, dgrid, table)`` from :func:`exact_inverse_table`
        (with ``sgrid`` the uniform grid of normalised ``σ²``), to reuse;
        default: built for the range of ``sigma / alpha``.

    Returns
    -------
    Tensor
    """
    alpha = torch.as_tensor(alpha, dtype=d.dtype, device=d.device)
    sigma = torch.as_tensor(sigma, dtype=d.dtype, device=d.device)
    s2 = (sigma / alpha).square()
    if method == "algebraic":
        y = 0.25 * d * d - 0.375 - s2
    elif method == "asymptotic":
        y = _asymptotic(d, s2)
    elif method == "exact":
        if table is None:
            table = build_table(s2)
        sgrid, dgrid, tab = (t.to(d) for t in table)
        y = torch.where(
            d > dgrid[-1],
            _asymptotic(d, s2),
            _lookup(d, s2, sgrid, dgrid, tab),
        )
    else:
        raise ValueError(f"Unknown method {method!r}")
    return alpha * y + mu


def build_table(s2: Tensor, n: int = 33) -> tuple[Tensor, Tensor, Tensor]:
    """Exact-inverse table for normalised variances in the range of ``s2``.

    Returns ``(sgrid, dgrid, table)``: ``sgrid`` is a uniform grid of
    ``σ²`` with ``n`` points (one when ``s2`` is constant).
    """
    lo, hi = float(s2.min()), float(s2.max())
    if hi - lo <= 1e-9 * max(1.0, hi):
        sgrid = torch.tensor([lo], dtype=torch.float64)
    else:
        sgrid = torch.linspace(lo, hi, n, dtype=torch.float64)
    dgrid, tab = exact_inverse_table(sgrid)
    return sgrid, dgrid, tab


class GeneralizedAnscombe(CalibrationEstimator):
    r"""Variance-stabilising transform for Poisson-Gaussian data.

    Model (Mäkitalo & Foi): :math:`z = \alpha\,\mathrm{Poisson}(y /
    \alpha) + \mathcal N(\mu, \sigma^2)`, i.e. a camera with gain
    :math:`\alpha` (ADU per electron), offset :math:`\mu` and read
    variance :math:`\sigma^2` (a :class:`SensorEstimate`).
    :meth:`apply` computes :func:`anscombe`, :meth:`inverse` its
    inverse.

    Parameters
    ----------
    alpha, sigma, mu : float, optional
        Noise parameters; alternatively ``fit(init=sensor)``.
    inverse : {"exact", "asymptotic", "algebraic"}
        Inverse used by :meth:`inverse`. Paper: ``"exact"``.
    per_pixel : bool
        Extension: use the per-pixel maps of the sensor estimate. Paper:
        False (scalar parameters: the means of the maps).
    faithful, dtype, device, layout
        See :class:`optix.Estimator`.

    Notes
    -----
    ``fit(Y=None, init=sensor)`` only reads the parameters (no data
    needed). The exact-inverse table is computed once (float64, CPU) and
    interpolated on the data device.
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
        self, Y: ArrayLike | None = None, **kwargs: object
    ) -> "GeneralizedAnscombe":
        """Read the noise parameters from ``init`` (a SensorEstimate).

        Parameters
        ----------
        Y : array-like, optional
            Unused.
        **kwargs
            ``init`` (required), ``layout``, ...

        Returns
        -------
        GeneralizedAnscombe
            self
        """
        return super().fit(Y, **kwargs)

    def _fit(
        self,
        Y: ArrayLike,
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
        self, Y: ArrayLike, out: WritableArrayLike | None
    ) -> WritableArrayLike:
        return self._apply_maps(Y, out, anscombe, *self._params())

    def inverse(
        self, D: ArrayLike, out: WritableArrayLike | None = None
    ) -> WritableArrayLike:
        """Inverse transform (method from the ``inverse`` option).

        Parameters
        ----------
        D : array-like
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

        if not hasattr(D, "layout"):
            D = self._labeled(D, self.layout)
        with torch.no_grad():
            return self._apply_maps(D, out, fn, alpha, sigma, mu)
