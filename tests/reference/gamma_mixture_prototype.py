import functools
import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
import torch
from torch import Tensor, as_tensor, get_default_device, get_default_dtype

"""
The physical model
------------------
s = Poisson rate of the "true" signal
b = Poisson rate of the "background" signal (autofluorescence)
d = Poisson rate of the "dark" signal (camera noise)
n = Photon count
m = Signal "modulation" (excitation efficiency x emission x sensitivity)
y = Image on the camera
o = Camera offset
γ = Camera gain
σ = Standard deviation of the camera read noise
i = Pixel index (y, x)
k = Frame index (tile, z)

nᵢₖ ~ Poisson(mᵢ·(sᵢₖ + bᵢ) + dᵢ)
yᵢₖ ~ Normal(γ·nᵢₖ + oᵢ, σᵢ²)

The reduced model
-----------------
u = γ·s               = The true signal in camera units
c = o  + γ ·(m·b + d) = The background signal in camera units
v = σ² + γ²·(m·b + d) = The variance in camera units

yᵢₖ ~ Normal(cᵢ + mᵢ·uᵢₖ, vᵢ + γ·mᵢ·uᵢₖ)

The statistical model
---------------------
z = Tissue class indicator
α = Tissue mixture concentrations
μ = Prior expected value of the gamma distributions
ν = Prior degrees of freedom of the gamma distributions
π = Tissue mixture proportions [per frame]
θ = Encoding of the log-modulation field in a DCT basis without DC term
c = Tissue class index
B = DCT basis (without DC term)
L = λ · diag(bending energy of each basis function)

πₖ  ~ Dirichlet(α)
zᵢₖ ~ Categorical(πₖ)
uᵢₖ | (zᵢₖ = background) ~ Dirac(0)
uᵢₖ | (zᵢₖ = tissue c)   ~ Gamma(ν[c], μ[c] / ν[c])
θ  ~ Normal(0, L⁻¹)
mᵢ = exp(Bᵢ · θ)

Inference
---------
The variational posterior is

    q = Πₖ q(πₖ) · Πᵢₖ q(zᵢₖ) q(uᵢₖ | zᵢₖ) · δ(c, v, θ, γ, μ, ν, α)

* q(z, u) is exact given everything else. The integral over u is done by
  Gauss-Hermite quadrature in t = log u, centred on the mode of the
  integrand (so one node would be the Laplace approximation).
* q(πₖ) is Dirichlet (conjugate update).
* μ, ν, α are point estimates with closed-form / 1D Newton updates.
* c, v, θ are point estimates updated by one joint Fisher-scoring step
  per iteration: the per-pixel unknowns (c, v) are eliminated exactly,
  which leaves a small dense system in θ. The gain is held at its
  nominal value by `fit` (see `update_gamma`).

Noise variants (`noise=`)
-------------------------
"full"     : Var[y | u] = v + γ·m·u            (the reduced model above)
"constant" : Var[y | u] = v                    (no signal shot noise; γ unused)
"""

# ======================================================================
#
#                   G L O B A L    O P T I O N S
#
# ======================================================================

NoiseModel = Literal["full", "constant"]

BATCH = 8               # Number of frames processed at once
DTYPE = None            # Preferred datatype (None = Pytorch default = float32)
DEVICE = None           # Preferred device (None = Pytorch default = cpu)
COMPILE = True          # Whether to compile the quadrature kernel with `torch.compile` (if available)
MISSING = object()


def _backend() -> dict[str, torch.device | torch.dtype]:
    dtype = DTYPE or get_default_dtype()
    device = DEVICE or get_default_device()
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Invalid dtype: {dtype}")
    if isinstance(device, str):
        device = torch.device(device)
    if not isinstance(device, torch.device):
        raise TypeError(f"Invalid device: {device}")
    return {"dtype": dtype, "device": device}


def _as_tensor(x) -> Tensor:
    """Convert an array-like (tensor, numpy, zarr slice, ...) to a tensor on the default backend."""
    backend = _backend()
    if isinstance(x, Tensor):
        return x.to(**backend)
    dtype = torch.empty(0, dtype=backend["dtype"]).numpy().dtype
    return torch.as_tensor(np.asarray(x).astype(dtype, copy=False), device=backend["device"])


def _batch(value: int | None = None) -> int:
    if value is not None and value is not MISSING:
        global BATCH
        BATCH = value
    return BATCH


def _dtype(value: torch.dtype | None = MISSING) -> int:
    if value is not MISSING:
        if isinstance(value, str):
            value = getattr(torch, value)
        if not isinstance(value, torch.dtype):
            raise TypeError(f"Invalid dtype: {value}")
        global _DTYPE
        _DTYPE = value
    return _DTYPE


def _device(value: torch.device | None = MISSING) -> int:
    if value is not MISSING:
        if isinstance(value, str):
            value = torch.device(value)
        if not isinstance(value, torch.device):
            raise TypeError(f"Invalid device: {value}")
        global _DEVICE
        _DEVICE = value
    return _DEVICE



def _compile(value: bool | None = None) -> bool:
    """Get (and optionally set) whether the quadrature kernel goes through `torch.compile`."""
    global COMPILE
    if value is not None:
        COMPILE = bool(value)
    return COMPILE


# Indices of the per-pixel statistics accumulated by the local step.
# All are posterior expectations, summed over frames, with
#   r  = y - c - m·u          (residual)
#   σ² = v + γ·m·ũ            (variance; ũ = u, plug-in value, or 0)
#   q  = (r² + δ - σ²) / (2 σ⁴)   (score with respect to the variance,
#                                  δ = quantization²/12, see `quantization`)
_HCC, _GC, _XU, _G1, _AU, _HVV, _GV, _WU, _G2, _BU = range(10)
# | 1/σ² | r/σ² | u/σ² | u·r/σ² | u²/σ² | 1/2σ⁴ | q | ũ/2σ⁴ | ũ·q | ũ²/2σ⁴ |


# ----------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------


@dataclass
class HyperPrior:
    """Prior over parameters and hyper-parameters fitted by VB or MAP."""

    alpha: Tensor
    """
    Tissue mixture concentrations, with shape (nc+1,) where `nc` is the
    number of tissue classes. The background class is last.
    """

    mu: Tensor
    """
    Prior expected value of the gamma distributions, with shape (nc,)
    where `nc` is the number of tissues classes.
    """

    nu: Tensor
    """
    Prior degrees of freedom of the gamma distribution, with shape (nc,)
    where `nc` is the number of tissues classes.
    """

    v: Tensor | None = None
    """
    Prior variance in an empty pixel (with no tissue).

    v = σ² + γ²(d + b·m)

    None: set to the median of the initial V (by `init`, or by the first
    local step if `init` is skipped).
    """

    v_dof: float = 5.0
    """
    Strength of the prior on V, as a number of pseudo-observations with
    squared residual `v` (inverse-gamma-like: log p(V) = -½·v_dof·(log V + v/V)).
    The fixed point for a pure-background pixel with N frames is
    V = (Σ (e² + δ) + v_dof·v) / (N + v_dof). 0 disables it.
    """

    lam: float = 1.0
    """
    Precision of the bending energy prior on the log-modulation field:
    L = lam · diag(bending_weights).
    """

    gain_sd: float = 0.1
    """
    Standard deviation of the log-normal prior on the gain, centred on
    its nominal (initial) value. Only used when the gain is estimated.
    """

    def __post_init__(self) -> None:
        backend = _backend()
        self.alpha = as_tensor(self.alpha, **backend).flatten()
        self.mu = as_tensor(self.mu, **backend).flatten()
        self.nu = as_tensor(self.nu, **backend).flatten()
        if self.v is not None:
            self.v = as_tensor(self.v, **backend)
        if (
            (len(self.alpha) != len(self.mu) + 1) or
            (len(self.alpha) != len(self.nu) + 1)
        ):
            raise ValueError("Number of tissue classes do not match")

    @classmethod
    def from_nclass(cls, nclass: int) -> Self:
        backend = _backend()
        # Note: alpha = 0 and nu = 0 are improper (ψ(0) = -inf, and a Gamma
        # with zero shape is degenerate), so the defaults are ones.
        alpha = torch.ones(nclass + 1, **backend)
        mu = torch.ones(nclass, **backend)
        nu = torch.ones(nclass, **backend)
        return cls(alpha=alpha, mu=mu, nu=nu)


@dataclass
class HyperPosterior:
    """Posterior over hyper-parameters fitted by VB."""

    alpha: Tensor
    """
    Posterior tissue mixture concentrations, with shape (*batch, frames, nc+1)
    where `nc` is the number of tissue classes. The background class is last.
    """

    mu: Tensor
    """
    Mean of the Gamma distribution that best fits the aggregated posterior
    of U in each tissue class, with shape (nc,). `update_prior` copies it
    into the prior (this is the M-step for μ).
    """

    nu: Tensor
    """
    Shape of the Gamma distribution that best fits the aggregated posterior
    of U in each tissue class, with shape (nc,). `update_prior` copies it
    into the prior (this is the M-step for ν).
    """

    def __post_init__(self) -> None:
        backend = _backend()
        self.alpha = as_tensor(self.alpha, **backend)
        self.mu = as_tensor(self.mu, **backend).flatten()
        self.nu = as_tensor(self.nu, **backend).flatten()
        if (
            (self.alpha.shape[-1] != len(self.mu) + 1) or
            (self.alpha.shape[-1] != len(self.nu) + 1)
        ):
            raise ValueError("Number of tissue classes do not match")

    @classmethod
    def from_shape(cls, shape: tuple[int, ...], nclass: int) -> Self:
        """shape : (*batch, frames)"""
        backend = _backend()
        alpha = torch.ones((*shape, nclass + 1), **backend)
        mu = torch.ones(nclass, **backend)
        nu = torch.ones(nclass, **backend)
        return cls(alpha=alpha, mu=mu, nu=nu)


@dataclass
class Parameters:
    """Parameters fitted by ML/MAP (on the variational bound)"""

    C: Tensor
    """
    The background signal in camera units, with shape (height, width).
    """

    V: Tensor
    """
    The background variance in camera units, with shape (height, width).
    """

    M: Tensor
    """
    The signal modulation, with shape (height, width).
    """

    theta: Tensor
    """
    The log-modulation field in a 2D DCT basis, with shape (ky, kx).
    The DC coefficient theta[0, 0] is always zero, so that the geometric
    mean of M is one.
    """

    gain: Tensor = 1.0
    """
    The camera gain (camera units per photoelectron), scalar.
    """

    def __post_init__(self) -> None:
        backend = _backend()
        self.C = as_tensor(self.C, **backend)
        self.V = as_tensor(self.V, **backend)
        self.M = as_tensor(self.M, **backend)
        self.theta = as_tensor(self.theta, **backend)
        self.gain = as_tensor(self.gain, **backend)

        if (
            (self.C.shape != self.V.shape) or
            (self.C.shape != self.M.shape) or
            (self.theta.ndim != 2)
        ):
            raise ValueError("Variable shapes do not match")

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        nclass: int,
        order: tuple[int, int] = (4, 4),
        gain: float = 1.0,
    ) -> Self:
        """shape : (..., height, width)"""
        backend = _backend()
        C = torch.zeros(shape[-2:], **backend)
        V = torch.ones(shape[-2:], **backend)
        M = torch.ones(shape[-2:], **backend)
        order = (min(order[0], shape[-2]), min(order[1], shape[-1]))
        theta = torch.zeros(order, **backend)
        return cls(C=C, V=V, M=M, theta=theta, gain=gain)


@dataclass
class GammaMixtureModel:

    Y: Tensor
    """
    Acquired data, with shape (*batch, frames, height, width), with any
    number (0, 1, 2+) of batch dimensions. Any array-like that supports
    `.shape` and slicing (tensor, numpy, zarr, ...); it is only converted
    to a tensor one batch of frames at a time (except in `init`).
    """

    nclass: int = 1
    prior: HyperPrior | None = None
    post: HyperPosterior | None = None
    param: Parameters | None = None

    noise: NoiseModel = "full"
    """Noise variant: "full", "constant" (see module docstring)."""

    order: tuple[int, int] = (8, 8)
    """Number of DCT basis functions per dimension for the log-modulation."""

    gain: float = 1.0
    """Nominal camera gain (camera units per photoelectron)."""

    nodes: int = 24
    """Number of Gauss-Hermite nodes for the integral over U."""

    nu_bounds: tuple[float, float] = (1.0, 1e3)
    """
    Bounds on the Gamma shapes. The lower bound of 1 prevents a tissue
    class from developing a spike at zero and competing with the
    background class.
    """

    quantization: float = 1.0
    """
    Width of one quantisation bin, in camera units (1 for integer data,
    0 to treat the data as continuous). Each observation is replaced by a
    uniform distribution over its bin, q(y*) = U(y - Δ/2, y + Δ/2), and the
    Gaussian log-likelihood by its expectation under q, which is a lower
    bound on log P(y falls in its bin):

        E_q[log N(y* | μ, s²)] + H[q] = log N(y | μ, s²) - Δ²/(24 s²) + log Δ

    i.e. every squared residual e² becomes e² + Δ²/12. This keeps the bound
    finite when a pixel takes the same value in many frames (otherwise the
    background variance can collapse to zero and the bound diverges).
    """

    history: list[float] = field(default_factory=list)
    """Lower bound after each local step."""

    Z: Tensor | None = field(default=None, init=False, repr=False)
    """
    Posterior class probabilities, computed by `classify` (at the end of
    `fit`). Either the foreground probability p(z ≠ background | y), with
    the shape of Y, or the per-tissue-class probabilities p(z = k | y),
    with shape (nclass, *Y.shape) (the background is 1 - Z.sum(0)).
    """

    U: Tensor | None = field(default=None, init=False, repr=False)
    """
    Posterior mean of the signal E[u | y] (camera units, with the
    modulation removed), with the shape of Y. Only computed by
    `classify(mean=True)`.
    """

    def __post_init__(self) -> None:
        if len(self.Y.shape) < 3:
            raise ValueError("Y must have shape (*batch, frames, height, width)")
        if self.noise not in ("full", "constant"):
            raise ValueError(f"Unknown noise model: {self.noise}")
        self._initialised = self.param is not None
        self._numel = math.prod(self.Y.shape)
        if self.prior is None:
            self.prior = HyperPrior.from_nclass(nclass=self.nclass)
        if self.post is None:
            self.post = HyperPosterior.from_shape(shape=self.Y.shape[:-2], nclass=self.nclass)
        if self.param is None:
            self.param = Parameters.from_shape(
                shape=self.Y.shape, nclass=self.nclass, order=self.order, gain=self.gain,
            )
        self.nclass = len(self.prior.mu)
        height, width = self.Y.shape[-2:]
        order = self.param.theta.shape
        self._by = dct_basis(height, order[0])
        self._bx = dct_basis(width, order[1])
        self._bending = bending_weights((height, width), order)
        gh_nodes, gh_logws = gauss_hermite(self.nodes)
        # Python floats: baked into the compiled kernel as constants
        self._gh = tuple(gh_nodes.tolist()), tuple(gh_logws.tolist())
        self._gain0 = self.param.gain.clone()
        if self.quantization < 0:
            raise ValueError("quantization must be non-negative")
        self._dq = self.quantization ** 2 / 12  # variance of one bin
        self._stats: Tensor | None = None
        self._counts: Tensor | None = None
        self._class_stats: tuple[Tensor, Tensor] | None = None

    # ------------------------------------------------------------------
    # Plug-in summaries (kept from the skeleton)
    # ------------------------------------------------------------------

    def modulation(self) -> Tensor:
        """Evaluate M = exp(B·θ) on the pixel grid."""
        return (self._by @ self.param.theta @ self._bx.T).exp()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _frame_batches(self, Y=None):
        """
        Iterate over batches of (at most `_batch()`) frames of `Y` (default:
        the training data), loading each lazily. Yields the index
        `(*batch_index, frame_slice)` and the data, with shape
        (frames, height, width).
        """
        Y = self.Y if Y is None else Y
        *batch, nframes, _, _ = Y.shape
        for chunk in itertools.product(*(range(s) for s in batch)):
            for start in range(0, nframes, _batch()):
                sl = (*chunk, slice(start, start + _batch()))
                yield sl, _as_tensor(Y[sl])

    def _row_tiles(self):
        """
        Iterate over tiles of rows with all frames of all batches, loading
        each lazily. Tiles have about as many elements as a batch of
        `_batch()` frames. Yields the row slice and the data, with shape
        (*batch, frames, rows, width).
        """
        *batch, nframes, height, _ = self.Y.shape
        nrows = max(1, (_batch() * height) // (math.prod(batch) * nframes))
        for start in range(0, height, nrows):
            rows = slice(start, start + nrows)
            yield rows, _as_tensor(self.Y[..., rows, :])

    @torch.no_grad()
    def init(
        self,
        bg_fraction: float = 0.5,
        threshold: float = 3.0,
        modulation: bool = True,
    ) -> None:
        """
        Data-driven starting point.

        bg_fraction : rough (lower bound on the) fraction of frames in
            which a pixel is empty. The lowest `bg_fraction` of the values
            of each pixel are assumed to be background: their median
            initialises C and their spread initialises V.
        threshold : number of standard deviations above C beyond which a
            value is initially labelled as tissue.
        modulation : initialise M from the mean tissue signal of each
            pixel (flat-field moment estimate, see `fit_modulation`);
            otherwise M = 1.
        """
        prm, prior, post = self.param, self.prior, self.post
        *batch, nframes, height, width = self.Y.shape
        nframes, nclass = math.prod(batch) * nframes, self.nclass
        backend = _backend()

        # Pass 1, over row tiles with all frames (order statistics need
        # every frame of a pixel): baseline and variance from the lower
        # order statistics of each pixel.
        k_med = min(max(round(0.5 * bg_fraction * nframes), 1), nframes)
        k_low = min(max(round(0.1587 * bg_fraction * nframes), 1), k_med)
        C = torch.empty((height, width), **backend)
        sd = torch.empty((height, width), **backend)
        abs_dev = []  # per-tile median |y - C| (fallback for sd_typical)
        for rows, y in self._row_tiles():
            y = y.reshape(-1, *y.shape[-2:])
            C[rows] = torch.kthvalue(y, k_med, dim=-3).values
            sd[rows] = C[rows] - torch.kthvalue(y, k_low, dim=-3).values
            abs_dev.append((y - C[rows]).abs_().median())
        sd_typical = sd.median()
        if not sd_typical > 0:
            sd_typical = torch.stack(abs_dev).median().clamp_min_(1e-3)
        sd = sd.clamp(0.25 * sd_typical, 4 * sd_typical)
        V = sd.square()

        # Pass 2, over batches of frames: initial labels and signal.
        # Every batch contributes about the same share of the (at most
        # `max_samples`) tissue values used to initialise the classes.
        max_samples = 1_000_000
        nbatches = math.prod(batch) * math.ceil(self.Y.shape[-3] / _batch())
        #
        # For M: per pixel, the number of tissue values and the sum of
        # their excess over the threshold. For an exponential signal,
        # E[r - t | r > t] = M·μ whatever the threshold t (memorylessness),
        # so the excess is not biased by the truncation where M is small.
        n_tissue = torch.empty(post.alpha.shape[:-1], **backend)
        cut = threshold * sd
        fg_count = torch.zeros((height, width), **backend)
        fg_excess = torch.zeros((height, width), **backend)
        moments = torch.zeros(3, dtype=torch.float64, device=C.device)
        values = []
        for sl, y in self._frame_batches():
            resid = y - C
            tissue = resid > cut
            n_tissue[sl] = tissue.sum((-2, -1)).to(y.dtype)
            vals = resid[tissue]
            step = max(1, vals.numel() * nbatches // max_samples)
            values.append(vals[::step])
            excess = resid.sub_(cut).mul_(tissue)  # resid is overwritten
            fg_count += tissue.sum(-3)
            fg_excess += excess.sum(-3)
            moments += torch.stack([
                tissue.sum(), excess.sum(), excess.square().sum(),
            ]).to(moments)
        values = torch.cat(values)

        theta = torch.zeros_like(prm.theta)
        if modulation and moments[0] > 0:
            n, s1, s2 = moments.tolist()
            mean = s1 / n
            # Information about log M per tissue value: mean² / variance
            # (= ν for a Gamma signal without noise)
            info = mean ** 2 / max(s2 / n - mean ** 2, 1e-12 * mean ** 2)
            theta = fit_modulation(
                fg_excess, fg_count, self._by, self._bx,
                prior.lam * self._bending, info=info,
            )
        prm.theta = theta
        prm.M = self.modulation()
        # The class initialisation below uses the raw tissue values (M is
        # close to 1, and the k-means is sensitive to small perturbations).
        n_tissue = n_tissue.reshape(-1)
        if values.numel() >= 2 * nclass:
            mu, nu, frac = init_gamma_classes(values, nclass)
        else:
            mu = threshold * sd_typical * torch.arange(2, nclass + 2, **_backend())
            nu = torch.ones(nclass, **_backend())
            frac = torch.full((nclass,), 1 / nclass, **_backend())
        nu = nu.clamp(self.nu_bounds[0], min(self.nu_bounds[1], 20.0))

        prior.mu, prior.nu = mu.clone(), nu.clone()
        post.mu, post.nu = mu.clone(), nu.clone()
        if prior.v is None:
            prior.v = V.median()

        counts = torch.cat(
            [n_tissue[:, None] * frac, (height * width - n_tissue)[:, None]], dim=1,
        )
        post.alpha = (prior.alpha + counts).reshape(post.alpha.shape)

        prm.C, prm.V = C, V
        self._initialised = True

    # ------------------------------------------------------------------
    # Local step: q(Z, U)
    # ------------------------------------------------------------------

    def _quadrature(
        self, Y: Tensor, mu: Tensor, nu: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Integrate Gamma(u | ν, μ/ν) · Normal(y | c + m·u, σ²(u)) over u
        for one tissue class and one batch of frames.

        Parameters
        ----------
        Y  : (batch, height, width) observed data
        mu : () prior mean of the gamma distribution
        nu : () prior shape of the gamma distribution

        Returns
        -------
        logE1 : (batch, height, width) log of the integral = log E_{u~q}[1 | z]
        EU    : (batch, height, width) E_{u~q}[u | z]
        ElogU : (batch, height, width) E_{u~q}[log u | z]
        Emom  : (10, batch, height, width) conditional expectations of the
                quantities listed next to the `_HCC ...` indices
        """
        prm = self.param
        nodes, logws = self._gh
        return _quadrature_kernel(
            Y, prm.C, prm.V, prm.M, prm.gain, mu, nu,
            dq=self._dq, nodes=nodes, logws=logws,
            full=self.noise == "full",
        )

    @torch.no_grad()
    def update_latent(self) -> float:
        """
        Update q(Z, U) for every pixel and frame, and accumulate the
        statistics that every other update needs. Returns the lower bound.

        The bound is

            Σᵢₖ log Σ_z exp(E[log πₖ_z]) · p(yᵢₖ | z)
            - Σₖ KL(q(πₖ) || p(πₖ)) + log p(θ) + log p(V) + log p(γ)
        """
        prm, prior, post = self.param, self.prior, self.post
        *batch, nframes, height, width = self.Y.shape
        nk = self.nclass
        C, V = prm.C, prm.V
        if prior.v is None:  # `init` was skipped
            prior.v = V.median()

        alpha = post.alpha
        elogpi =torch.digamma(alpha) - torch.digamma(alpha.sum(-1, keepdim=True))

        self._stats, self._counts, self._class_stats = None, None, None

        stats = C.new_zeros((10, height, width))
        counts = C.new_zeros((*batch, nframes, nk + 1))
        sum_u = C.new_zeros(nk)
        sum_logu = C.new_zeros(nk)
        bound = 0.0

        for sl, y in self._frame_batches():
            st, bnd, cnt, u, logu = self._update_latent_chunk(y, elogpi[sl])
            stats += st
            counts[sl] += cnt
            sum_u += u
            sum_logu += logu
            bound += bnd

        self._stats = stats
        self._counts = counts
        self._class_stats = (sum_u, sum_logu)

        # Terms of the bound that do not depend on the data
        if self.quantization > 0:
            bound += self._numel * math.log(self.quantization)  # H[q(y*)]
        bound -= kl_dirichlet(alpha, prior.alpha).item()
        bound -= 0.5 * prior.lam * (self._bending * prm.theta.square()).sum().item()
        if prior.v_dof > 0:
            bound -= 0.5 * prior.v_dof * V.log().add_(V.reciprocal(), alpha=prior.v).sum().item()
        bound -= 0.5 * ((prm.gain / self._gain0).log().item() / prior.gain_sd) ** 2
        self.history.append(bound)
        return bound

    def _update_latent_chunk(self, Y, elogpi):
        # Parameters
        # ----------
        # Y : (frames, height, width)
        # elogpi : (frames, classes)
        #
        # Returns
        # -------
        # stats : (10, height, width) sufficient statistics for the M-step
        # bound : float lower bound on the log-likelihood
        # sum_Z : (frames, classes+1) expected counts of each class
        # sum_U : (classes,) expected sum of U in each tissue class
        # sum_logU : (classes,) expected sum of log U in each tissue class

        C = self.param.C
        V = self.param.V
        nk = self.nclass
        mu = self.prior.mu
        nu = self.prior.nu
        dq = self._dq

        # Background class: u = 0, variance V
        R0 = Y - C
        logZ = (
            R0.square().add_(dq).div_(V).add_(V.log()).mul_(-0.5)
            .add_(elogpi[:, -1, None, None] - 0.5 * math.log(2 * math.pi))
        )
        acc = Y.new_zeros((10, *Y.shape))
        acc[_HCC].copy_(V).reciprocal_()
        acc[_GC].copy_(R0).div_(V)
        acc[_HVV].copy_(V).square_().reciprocal_().mul_(0.5)
        acc[_GV].copy_(R0).square_().add_(dq).sub_(V).div_(V.square()).mul_(0.5)
        logjoint = [None] * (nk + 1)
        logjoint[nk] = logZ

        # Tissue classes, folded in with a running log-sum-exp so that
        # the per-class statistics never need to be stored.
        EUs, ElogUs = [], []
        for k in range(nk):
            logZ0 = logZ
            logpY, EU, ElogU, st = self._quadrature(Y, mu[k], nu[k])    # log p(y | z=k)
            logZY = elogpi[:, k, None, None] + logpY                    # log p(y, z=k)
            logZ = torch.logaddexp(logZ, logZY)                         # log Σ_k p(y, z=k)
            pz_prv = (logZ0 - logZ).exp_()
            pz_new = (logZY - logZ).exp_()
            acc.mul_(pz_prv).add_(st.mul_(pz_new))  # E[stat] = Σ_k p(z=k | y) E[stat | z=k]
            logjoint[k] = logZY
            EUs.append(EU)
            ElogUs.append(ElogU)

        sum_Z = Y.new_zeros((len(Y), nk + 1))
        sum_U = Y.new_zeros(nk)
        sum_logU = Y.new_zeros(nk)
        for k in range(nk + 1):
            Zk = (logjoint[k] - logZ).exp_()
            sum_Z[:, k] = Zk.sum((-2, -1))
            if k < nk:
                EU = EUs.pop(0)
                ElogU = ElogUs.pop(0)
                sum_U[k] = (Zk * EU).sum()
                sum_logU[k] = (Zk * ElogU).sum()

        stats = acc.sum(-3)  # sum over frames
        bound = logZ.sum().item()
        return stats, bound, sum_Z, sum_U, sum_logU

    @torch.no_grad()
    def classify(
        self,
        per_class: bool = False,
        mean: bool = False,
        out: object | None = None,
        out_mean: object | None = None,
    ) -> Tensor:
        """
        Compute the posterior class probabilities of every pixel and frame
        of the training data under the current parameters and q(π), and
        store them in `self.Z` (and the posterior mean of u in `self.U`).

        per_class : if False, return the foreground probability
            p(z ≠ background | y), with the shape of Y. If True, return
            p(z = k | y) for each tissue class, with shape (nclass, *Y.shape);
            the background probability is 1 - Z.sum(0).
        mean : also compute the posterior mean E[u | y] (the background
            class contributes u = 0).
        out, out_mean : optional preallocated outputs (tensor, numpy, zarr,
            or anything that supports slice assignment), e.g. to write
            results that do not fit in memory to disk. By default, tensors
            are allocated on the current backend.

        See `predict` for new frames.
        """
        Z, U, _ = self._posterior(
            self.Y, self.post.alpha, per_class, mean, out, out_mean, niter=0,
        )
        self.Z, self.U = Z, U
        return Z

    @torch.no_grad()
    def predict(
        self,
        Y: object,
        per_class: bool = False,
        mean: bool = False,
        out: object | None = None,
        out_mean: object | None = None,
        niter: int = 20,
        tol: float = 1e-3,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """
        Classify new frames with the learned model (C, V, M, γ, μ, ν, α
        are kept fixed). Only the per-frame proportions q(π) are inferred,
        by alternating q(Z) and q(π) for each batch of frames. The class
        likelihoods p(y | z) do not depend on π, so they are computed once
        per batch and the iterations cost almost nothing: the price is
        the same as one pass of `update_latent`.

        Y : new data, with shape (*batch, frames, height, width), any
            array-like (the image size must match the training data)
        per_class, mean, out, out_mean : see `classify`
        niter : maximum number of q(Z)/q(π) iterations per batch
        tol : stop when the posterior class counts of every frame change
            by less than `tol` (as a fraction of the number of pixels)

        Returns Z, U (None unless `mean`), and the posterior Dirichlet
        concentrations of each frame, with shape (*batch, frames, nclass+1).
        """
        if tuple(Y.shape[-2:]) != tuple(self.Y.shape[-2:]):
            raise ValueError("New frames must have the same size as the training data")
        alpha = self.prior.alpha.expand((*Y.shape[:-2], self.nclass + 1)).clone()
        return self._posterior(Y, alpha, per_class, mean, out, out_mean, niter, tol)

    def _posterior(self, Y, alpha, per_class, mean, out, out_mean, niter, tol=0.0):
        nk = self.nclass
        shape = (nk, *Y.shape) if per_class else tuple(Y.shape)
        if out is None:
            out = torch.empty(shape, **_backend())
        if mean and out_mean is None:
            out_mean = torch.empty(tuple(Y.shape), **_backend())
        npix = Y.shape[-1] * Y.shape[-2]

        for sl, y in self._frame_batches(Y):
            logbg, logfg, EU = self._class_loglik(y, mean)
            a = alpha[sl]
            for _ in range(niter):  # q(π) <-> q(Z), with fixed likelihoods
                elogpi = torch.digamma(a) - torch.digamma(a.sum(-1, keepdim=True))
                lbg = logbg + elogpi[:, -1, None, None]
                lfg = logfg + elogpi[:, :-1].T[..., None, None]
                logZ = torch.logaddexp(torch.logsumexp(lfg, 0), lbg)
                counts = torch.cat([
                    (lfg - logZ).exp_().sum((-2, -1)).T,
                    (lbg - logZ).exp_().sum((-2, -1))[:, None],
                ], -1)
                a_new = self.prior.alpha + counts
                done = (a_new - a).abs().max() < tol * npix
                a = a_new
                if done:
                    break
            alpha[sl] = a

            # Final responsibilities
            elogpi = torch.digamma(a) - torch.digamma(a.sum(-1, keepdim=True))
            logbg.add_(elogpi[:, -1, None, None])
            logfg.add_(elogpi[:, :-1].T[..., None, None])
            logtissue = torch.logsumexp(logfg, 0)
            logZ = torch.logaddexp(logtissue, logbg)
            # Computed from the tissue terms directly (not as 1 - p(bg)) so
            # that small foreground probabilities keep their precision.
            if per_class:
                z = logfg.sub_(logZ).exp_()
                _store(out, (slice(None), *sl), z)
                if mean:
                    _store(out_mean, sl, z.mul_(EU).sum(0))
            else:
                _store(out, sl, logtissue.sub_(logZ).exp_())
                if mean:
                    _store(out_mean, sl, logfg.sub_(logZ).exp_().mul_(EU).sum(0))

        return out, (out_mean if mean else None), alpha

    def _class_loglik(self, Y, mean):
        # Y : (frames, height, width)
        # Returns log p(y | background), log p(y | z = k) with shape
        # (classes, frames, height, width), and E[u | y, z = k] (or None).
        prm, prior = self.param, self.prior
        V = prm.V
        logbg = (
            (Y - prm.C).square_().add_(self._dq).div_(V).add_(V.log())
            .add_(math.log(2 * math.pi)).mul_(-0.5)
        )
        logfg, EU = [], []
        for k in range(self.nclass):
            logpY, EUk, _, _ = self._quadrature(Y, prior.mu[k], prior.nu[k])
            logfg.append(logpY)
            if mean:
                EU.append(EUk)
        return logbg, torch.stack(logfg), (torch.stack(EU) if mean else None)

    # ------------------------------------------------------------------
    # Parameter step: C, V, θ, γ
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_parameters(
        self,
        c: bool = True,
        v: bool = True,
        theta: bool = True,
        gain: bool = False,
        lr: float = 1.0,
        max_log_step: float = 0.5,
    ) -> None:
        """
        One joint Fisher-scoring step on the selected parameters, using the
        statistics of the last local step (run `update_latent` first).

        The complete-data Fisher information is block-diagonal across
        pixels in (C, V), and couples them to (θ, log γ) only through
        per-pixel terms. C and V are therefore eliminated analytically
        (Schur complement), which leaves a dense system of the size of the
        basis for θ and a scalar equation for log γ, and are then
        back-substituted.

        With `noise="constant"` and M fixed, the C and V steps are the
        exact closed-form EM updates.
        """
        if self._stats is None:
            raise RuntimeError("Run update_latent() before updating parameters")
        prm, prior = self.param, self.prior
        Hcc, Gc, Hcm, Gm, Hmm, Hvv, Gv, Hvm, Gmg, Hmg = self._stats.unbind(0)
        M, V, gam = prm.M, prm.V, prm.gain
        gain = gain and self.noise != "constant"

        if prior.v_dof > 0:
            hvp = V.square().reciprocal_().mul_(prior.v_dof * 0.5)
            gvp = (prior.v - V).mul_(hvp)
            Gv = Gv + gvp
            Hvv = Hvv + hvp

        gM = gam * M
        Fcm = M * Hcm                    # Fisher(C, log M)
        Fvm = gM * Hvm                   # Fisher(V, log M) = Fisher(V, log γ)
        Fmm = M.square().mul_(Hmm)       # Fisher(log M, log M), mean part
        Vmm = gM.square().mul_(Hmg)      # Fisher(log M, log M), variance part
        Sm = M.mul(Gm).addcmul(gM, Gmg)  # score of log M
        Sg = gM * Gmg                    # score of log γ (per pixel)
        tiny = torch.finfo(M.dtype).tiny
        iHcc = Hcc.clamp_min(tiny).reciprocal_()
        iHvv = Hvv.clamp_min(tiny).reciprocal_()

        # Eliminate C and V
        Fmm += Vmm
        if c:
            Fmm.sub_(Fcm.square().mul_(iHcc))
            Sm.sub_(Fcm.mul(Gc).mul_(iHcc))
        if v:
            tmp = Fvm.square().mul_(iHvv)
            Fmm.sub_(tmp)
            Vmm.sub_(tmp)
            tmp = Fvm.mul(Gv).mul_(iHvv)
            Sm.sub_(tmp)
            Sg.sub_(tmp)
        Fmm.clamp_min_(0)
        Vmm.clamp_min_(0)

        # Reduced system in θ
        dlogm = torch.zeros_like(M)
        dtheta = torch.zeros_like(prm.theta)
        if theta:
            By, Bx = self._by, self._bx
            n = prm.theta.numel()
            weights = prior.lam * self._bending.flatten()
            tmp = torch.einsum("yx,xb,xd->ybd", Fmm, Bx, Bx)
            H = torch.einsum("ya,yc,ybd->abcd", By, By, tmp).reshape(n, n)
            H.diagonal().add_(weights)
            H.diagonal().add_(H.diagonal().mean(), alpha=1e-8)
            g = (By.T @ Sm @ Bx).flatten()
            g.addcmul_(weights, prm.theta.flatten(), value=-1)
            # No DC term: the geometric mean of M stays at one
            H[0, :] = H[:, 0] = 0
            H[0, 0] = 1
            g[0] = 0
            dtheta = torch.linalg.solve(H, g).reshape(prm.theta.shape)
            dlogm = By @ dtheta @ Bx.T
            shrink = min(1.0, max_log_step / max(dlogm.abs().max().item(), 1e-12))
            dtheta *= shrink
            dlogm *= shrink

        # Gain: scalar step on log γ, with V eliminated
        dloggain = 0.0
        if gain:
            deviation = (gam / self._gain0).log().item()
            score = Sg.sum().item() - deviation / prior.gain_sd ** 2
            curvature = Vmm.sum().item() + 1 / prior.gain_sd ** 2
            dloggain = max(-max_log_step, min(max_log_step, score / curvature))

        # Back-substitution
        if c:
            prm.C.add_(Gc.addcmul(Fcm, dlogm, value=-1).mul_(iHcc), alpha=lr)
        if v:
            dV = (dlogm + dloggain).mul_(Fvm).sub_(Gv).neg_().mul_(iHvv)
            prm.V = torch.maximum(prm.V.add(dV, alpha=lr), prm.V.mul(0.1))
        if theta:
            prm.theta.add_(dtheta, alpha=lr)
            prm.M = self.modulation()
        if gain:
            prm.gain = gam * math.exp(lr * dloggain)

    def update_c(self) -> None:
        self.update_parameters(c=True, v=False, theta=False, gain=False)

    def update_v(self) -> None:
        self.update_parameters(c=False, v=True, theta=False, gain=False)

    def update_theta(self) -> None:
        self.update_parameters(c=False, v=False, theta=True, gain=False)

    def update_gamma(self) -> None:
        """
        One step on the camera gain γ (not the Gamma distributions).

        Experimental, and deliberately not called by `fit`. In this model
        the gain is identified only through the shape of the signal
        distribution, and on simulated data the bound was almost flat in γ
        (a few log units over a threefold range), so free estimates drift.
        Prefer `gain=` from the camera specification. If you do call this,
        do it only once everything else has converged, and keep
        `prior.gain_sd` small so that γ stays near its nominal value.
        """
        self.update_parameters(c=False, v=False, theta=False, gain=True)

    # ------------------------------------------------------------------
    # Mixture step: q(π), then μ, ν, α
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_post(self) -> None:
        """
        Update q(πₖ) = Dirichlet(α + Σᵢ Zᵢₖ), and fit a Gamma distribution
        to the aggregated posterior of U in each tissue class.
        """
        if self._counts is None:
            raise RuntimeError("Run update_latent() before update_post()")
        prior, post = self.prior, self.post
        post.alpha = (prior.alpha + self._counts).reshape(post.alpha.shape)

        sum_u, sum_logu = self._class_stats
        nclass = self.nclass
        weight = self._counts[..., :-1].reshape(-1, nclass).sum(0)
        ok = weight > 1e-3
        safe = weight.clamp_min(1e-3)
        mu = (sum_u / safe).clamp_min_(torch.finfo(sum_u.dtype).tiny)
        nu = fit_gamma_shape(mu.log().sub_(sum_logu / safe)).clamp_(*self.nu_bounds)
        post.mu = torch.where(ok, mu, post.mu)
        post.nu = torch.where(ok, nu, post.nu)

    @torch.no_grad()
    def update_prior(self, alpha: bool = True) -> None:
        """
        Update the parameters of the prior distributions by maximising the
        bound: (μ, ν) of the Gamma distributions, and optionally the
        Dirichlet concentrations α.
        """
        prior, post = self.prior, self.post
        prior.mu = post.mu.clone()
        prior.nu = post.nu.clone()
        if alpha:
            a = post.alpha.reshape(-1, post.alpha.shape[-1])
            if len(a) > 1:
                elogpi = torch.digamma(a) - torch.digamma(a.sum(-1, keepdim=True))
                prior.alpha = fit_dirichlet(elogpi, prior.alpha)

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(
        self,
        niter: int = 100,
        tol: float = 1e-6,
        learn_modulation: bool = True,
        learn_alpha: bool = True,
        lr: float = 1.0,
        max_halvings: int = 4,
        init: bool | None = None,
        verbose: int = False,
        classify: bool = True,
        per_class: bool = False,
    ) -> Self:
        """
        Fit the model.

        niter : maximum number of iterations
        tol : stop when the bound improves by less than `tol` per data point
        learn_modulation : estimate the modulation field M
        learn_alpha : estimate the Dirichlet concentrations
        lr : damping of the Fisher-scoring step
        max_halvings : if the bound decreases, the last parameter step is
            halved up to this many times
        init : run the data-driven initialisation (default: only if no
            `param` was provided and `init` has not been called)
        classify : compute the final class probabilities (`self.Z`)
        per_class : store per-tissue-class probabilities instead of the
            foreground probability (see `classify`)
        """
        if init or (init is None and not self._initialised):
            self.init(modulation=learn_modulation)
        prm = self.param
        slack = tol * self._numel
        backtrack = max_halvings

        previous = None
        last = -math.inf
        for n in range(niter):
            bound = self.update_latent()

            halvings = 0
            while previous is not None and bound < last - slack and halvings < backtrack:
                halvings += 1
                halve_(prm.C, previous[0])
                halve_(prm.V, previous[1])
                halve_(prm.theta, previous[2])
                loghalve_(prm.gain, previous[3])
                prm.M = self.modulation()
                self.history.pop()
                bound = self.update_latent()

            if verbose:
                print(
                    f"{n:3d} | bound = {bound/self._numel:12.6e} | "
                    f"gain = {prm.gain.item():.4g} | "
                    f"mu = [{','.join([f'{round(x, 3):5.3f}' for x in self.prior.mu.tolist()])}] | "
                    f"nu = [{','.join([f'{round(x, 3):5.3f}' for x in self.prior.nu.tolist()])}] | "
                    f"c = {round(prm.C.mean().item(), 3):7.3f} | "
                    f"v = {round(prm.V.mean().item(), 3):7.3f} | "
                    f"m = {round(prm.M.mean().item(), 3)}"
                    + (f" | halved x{halvings}" if halvings else "")
                )
                if verbose > 1:
                    self.plot()
            if abs(bound - last) < slack:
                break  # q(Z, U) is already consistent with the parameters
            last = bound

            previous = list(map(torch.Tensor.clone, (prm.C, prm.V, prm.theta, prm.gain)))
            self.update_post()
            self.update_prior(alpha=learn_alpha)
            self.update_parameters(theta=learn_modulation, lr=lr)

        else:
            # Make the statistics and the bound consistent with the final
            # parameters (the prior/posterior were updated after the last
            # local step)
            self.update_latent()

        if classify:
            self.classify(per_class=per_class)
        return self

    def plot(self):
        import matplotlib.pyplot as plt

        fig = getattr(self, "_fig", None)
        new = fig is None or not plt.fignum_exists(fig.number)  # never opened, or closed
        if new:
            fig = plt.figure(figsize=(12, 4))
        self._fig = fig

        fig.clf()
        fig.set_layout_engine("constrained")

        ax = fig.add_subplot(1, 3, 1)
        im = ax.imshow(self.param.C.cpu())
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Offset")
        fig.colorbar(im, ax=ax)

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(self.param.V.cpu())
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Variance")
        fig.colorbar(im, ax=ax)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(self.param.M.cpu())
        ax.set_title("Modulation")
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax)

        # The window is only repainted while the GUI event loop runs, and
        # `fit` goes straight back to computing: process pending events now.
        # (`plt.pause` would also work, but raises the window every time.)
        if new:
            plt.show(block=False)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


def _store(out, index, value: Tensor) -> None:
    """Write a tensor into a tensor or any array-like (numpy, zarr, ...)."""
    if isinstance(out, Tensor):
        out[index] = value.to(out)
    else:
        out[index] = value.cpu().numpy()


# ----------------------------------------------------------------------
# Line search helpers
# ----------------------------------------------------------------------


def halve_(current, previous):
    current.mul_(0.5)
    current.add_(previous, alpha=0.5)
    return current


def loghalve_(current, previous):
    current.mul_(previous)
    current.sqrt_()
    return current


# ----------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------


def gauss_hermite(n: int) -> tuple[Tensor, Tensor]:
    """
    Gauss-Hermite rule (weight exp(-x²)) by the Golub-Welsch algorithm.

    Returns the nodes `x` and `log(w) + x²`, which is what is needed to
    integrate a function that is not explicitly multiplied by exp(-x²).
    """
    k = torch.arange(1, n, dtype=torch.float64, device="cpu")
    off = (k / 2).sqrt()
    jacobi = torch.diag(off, 1) + torch.diag(off, -1)
    x, vec = torch.linalg.eigh(jacobi)
    logw = 0.5 * math.log(math.pi) + 2 * vec[0].abs().log() + x.square()
    backend = _backend()
    return x.to(**backend), logw.to(**backend)


def dct_basis(n: int, k: int) -> Tensor:
    """Orthonormal DCT-II basis, shape (n, k). Column 0 is the DC term."""
    x = (torch.arange(n, dtype=torch.float64, device="cpu") + 0.5) / n
    f = torch.arange(k, dtype=torch.float64, device="cpu")
    basis = torch.cos(math.pi * x[:, None] * f[None, :]) * math.sqrt(2 / n)
    basis[:, 0] = math.sqrt(1 / n)
    return basis.to(**_backend())


def bending_weights(shape: tuple[int, int], order: tuple[int, int]) -> Tensor:
    """
    Bending energy of each (unit-norm) DCT basis function, in voxel units.

    The DCT-II functions are the eigenvectors of the discrete Laplacian
    with Neumann boundaries, with eigenvalue -(λy + λx) where
    λ = 2 - 2 cos(π k / N) ≈ (π k / N)². The bending energy Σᵢ (Δf)ᵢ² of
    f = B·θ is therefore Σ θ² (λy + λx)²: the *precision* goes as k⁴
    (coefficients weighted by k² inside the square), and in 2D the weight
    is (ky² + kx²)², not ky⁴ + kx⁴.
    """
    ky = torch.arange(order[0], dtype=torch.float64, device="cpu")
    kx = torch.arange(order[1], dtype=torch.float64, device="cpu")
    ly = 2 - 2 * torch.cos(math.pi * ky / shape[-2])
    lx = 2 - 2 * torch.cos(math.pi * kx / shape[-1])
    return (ly[:, None] + lx[None, :]).square().to(**_backend())


def fit_modulation(
    total: Tensor,
    count: Tensor,
    By: Tensor,
    Bx: Tensor,
    penalty: Tensor,
    info: float = 1.0,
    min_count: float = 3.0,
    robust_iter: int = 2,
    outlier: float = 4.0,
) -> Tensor:
    """
    Flat-field moment estimate of the log-modulation field.

    If the tissue signal has the same distribution in every pixel (on
    average over frames, and given that the pixel contains tissue), the
    mean tissue signal of pixel i is M_i·E[u]. Its log is fitted by
    weighted, penalised least squares in the DCT basis,

        min ½ Σᵢ wᵢ (log m̂ᵢ - (B·θ)ᵢ)² + ½ θᵀ·diag(penalty)·θ,

    where m̂ᵢ = total_i / count_i, and the weights wᵢ = info · count_i
    are the information about log M_i (info ≈ mean² / variance of a
    single tissue value). Pixels with fewer than `min_count` tissue
    values are ignored. The DC coefficient is
    then dropped (geometric mean of M = 1). Pixels whose residual exceeds
    `outlier` robust standard deviations (hot pixels, ...) are discarded
    and the fit repeated `robust_iter` times.

    total : (height, width) sum of tissue values of each pixel
    count : (height, width) number of tissue values of each pixel
    By, Bx : (height, ky), (width, kx) DCT bases
    penalty : (ky, kx) prior precision of θ

    Returns θ, with shape (ky, kx).
    """
    ky, kx = By.shape[-1], Bx.shape[-1]
    n = ky * kx
    ok = (count >= min_count) & (total > 0)
    if not ok.any():
        return torch.zeros((ky, kx), dtype=total.dtype, device=total.device)
    tiny = torch.finfo(total.dtype).tiny
    logm = torch.where(ok, total.clamp_min(tiny).log() - count.clamp_min(1).log(), 0)
    w = torch.where(ok, count * info, 0)
    for it in range(robust_iter + 1):
        tmp = torch.einsum("yx,xb,xd->ybd", w, Bx, Bx)
        H = torch.einsum("ya,yc,ybd->abcd", By, By, tmp).reshape(n, n)
        H.diagonal().add_(penalty.flatten())
        H.diagonal().add_(H.diagonal().mean(), alpha=1e-8)
        g = (By.T @ (w * logm) @ Bx).flatten()
        theta = torch.linalg.solve(H, g).reshape(ky, kx)
        if it == robust_iter:
            break
        res = (logm - By @ theta @ Bx.T).abs_()
        scale = 1.4826 * res[w > 0].median()
        w = torch.where(res > outlier * scale, 0, w)
    theta[0, 0] = 0
    return theta


def fit_gamma_shape(c: Tensor, niter: int = 16) -> Tensor:
    """Solve log(ν) - ψ(ν) = c for ν, with c > 0 (Minka's Newton scheme)."""
    c = c.clamp_min(1e-8)
    nu = (3 - c + ((c - 3).square() + 24 * c).sqrt()) / (12 * c)
    for _ in range(niter):
        f = nu.log() - torch.digamma(nu) - c
        df = 1 / nu - torch.polygamma(1, nu)
        nu = 1 / (1 / nu + f / (nu.square() * df)).clamp_min(1e-12)
    return nu


def init_gamma_classes(
    values: Tensor, nclass: int, niter: int = 50,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Initial Gamma classes from positive samples of the signal.

    The signal is heavy-tailed (most tissue is dim, a small fraction is
    very bright), so quantiles of the values all fall in the dim bulk.
    Instead, run a 1D k-means on log(values), with centres initially spread
    evenly in log between the 1st and 99.9th percentiles, so that a bright
    minority can claim its own class.

    Returns the mean, shape (moment-matched) and fraction of each class,
    sorted by increasing mean.
    """
    eps = torch.finfo(values.dtype).eps
    tiny = torch.finfo(values.dtype).tiny
    logv = values.clamp_min(tiny).log_()
    if nclass == 1:
        label = torch.zeros_like(logv, dtype=torch.long)
    else:
        lims = torch.quantile(logv, logv.new_tensor([0.01, 0.999]))
        centres = torch.linspace(lims[0].item(), lims[1].item(), nclass, **_backend())
        for _ in range(niter):
            label = (logv[:, None] - centres).abs_().argmin(1)
            count = torch.bincount(label, minlength=nclass).to(logv.dtype)
            total = torch.zeros_like(centres).index_add_(0, label, logv)
            new = torch.where(count > 0, total / count.clamp_min(1), centres)
            if torch.allclose(new, centres):
                break
            centres = new
        label = (logv[:, None] - centres).abs_().argmin(1)
    count = torch.bincount(label, minlength=nclass).to(values.dtype)
    total = torch.zeros(nclass, **_backend()).index_add_(0, label, values)
    total2 = torch.zeros(nclass, **_backend()).index_add_(0, label, values.square())
    n = count.clamp_min(1)
    mu = total / n
    mu2 = mu.square()
    var = (total2 / n).sub_(mu2).clamp_min_(eps * mu2)
    nu = torch.where(count > 1, mu2 / var, torch.ones_like(mu))
    # Empty classes: put them above the others
    mu = torch.where(count > 0, mu, values.max() * (1 + torch.arange(nclass, **_backend())))
    frac = (count + 1e-3) / (count.sum() + 1e-3 * nclass)
    order = mu.argsort()
    return mu[order], nu[order], frac[order]


def fit_dirichlet(
    elogpi: Tensor,
    alpha: Tensor,
    niter: int = 32,
    bounds: tuple[float, float] = (1e-3, 1e3),
) -> Tensor:
    """
    Maximise Σₖ E[log Dirichlet(πₖ | α)] with respect to α (Minka's Newton).

    elogpi : (frames, classes) expected log proportions under q(πₖ)
    alpha  : (classes,) starting point
    """
    nframes = elogpi.shape[-2]
    stat = elogpi.sum(0)
    alpha = alpha.clamp(*bounds)
    for _ in range(niter):
        total = alpha.sum()
        grad = nframes * (torch.digamma(total) - torch.digamma(alpha)) + stat
        diag = -nframes * torch.polygamma(1, alpha)
        const = nframes * torch.polygamma(1, total)
        shift = (grad / diag).sum() / (1 / const + (1 / diag).sum())
        step = (grad - shift) / diag
        scale = 1.0
        while ((alpha - scale * step) <= 0).any():
            scale /= 2
        alpha = (alpha - scale * step).clamp(*bounds)
    return alpha


def kl_dirichlet(a: Tensor, b: Tensor) -> Tensor:
    """KL(Dirichlet(a) || Dirichlet(b)), summed over leading dimensions."""
    a0 = a.sum(-1, keepdim=True)
    b0 = b.sum(-1, keepdim=True)
    kl = (
        torch.lgamma(a0).squeeze(-1) - torch.lgamma(a).sum(-1)
        - torch.lgamma(b0).squeeze(-1) + torch.lgamma(b).sum(-1)
        + ((a - b) * (torch.digamma(a) - torch.digamma(a0))).sum(-1)
    )
    return kl.sum()


# ----------------------------------------------------------------------
# Quadrature over U
# ----------------------------------------------------------------------


def _cubic_mode(
    R0: Tensor, V: Tensor, M: Tensor, gain: Tensor, nu: Tensor, beta: Tensor,
    dq: float, polish: int = 3,
) -> Tensor:
    """
    Mode û of the integrand over u (in t = log u) for the "full" noise model.

    With c = γm, r = r₀ - m·u and σ² = v + c·u, multiplying
        l'(t) = ν - β·u + u·[m·r/σ² + ½c·(r² + δ - σ²)/σ⁴]
    by σ⁴ > 0 gives a cubic P(u) = c₃u³ + c₂u² + c₁u + c₀ with
        c₃ = -βc² - ½m²c                                  < 0
        c₂ = νc² - 2βvc - m²v - ½c²
        c₁ = 2νvc - βv² + m·r₀·v + ½c·(r₀² + δ - v)
        c₀ = νv²                                          > 0
    Since P(0) > 0 > P(+∞), it has one positive root or three. With three
    (bimodal integrand), the smallest and largest are maxima of l and the
    middle one is a minimum: the maximum with the larger l is kept.

    The roots are computed in closed form (Cardano / Viète), with a fixed
    cost and without synchronisation, then polished by a few Newton steps
    on P (the closed form loses precision in float32).
    """
    # Inputs (R0, V, M, gain, nu, beta) are never modified: every in-place
    # operation below acts on a temporary created in this function. Shapes:
    # V, M and everything derived from them only are (H, W); anything that
    # involves R0 is (batch, H, W), and in-place ops keep the larger shape
    # on the left.
    tiny = torch.finfo(R0.dtype).tiny
    c = M * gain                                              # γm
    M2 = M.square()
    Vc = V * c
    c3 = (beta * c).add_(M2, alpha=0.5).mul_(c).neg_()        # -(βc + ½m²)·c
    c2 = (                                                    # (ν - ½)c² - 2βvc - m²v
        c.square().mul_(nu - 0.5)
        .sub_(Vc.mul(2 * beta))
        .sub_(M2.mul_(V))                                     # M2 is not needed after this
    )
    c1 = (                                                    # ½c(r₀² + δ - v) + m·r₀·v + 2νvc - βv²
        R0.square().add_(dq).sub_(V).mul_(c).mul_(0.5)
        .addcmul_(R0, M * V)
        .add_(Vc.mul_(2 * nu).sub_(V.square().mul_(beta)))    # Vc is not needed after this
    )
    c0 = V.square().mul_(nu)                                  # νv²

    # Monic form u³ + a·u² + b·u + e, depressed with u = x - a/3
    a3 = c2.div(c3).div_(3)                                   # a/3
    b = c1.div(c3)
    e = c0.div(c3)
    p = b.addcmul(a3, a3, value=-3)                           # b - a²/3
    q = b.neg_().addcmul_(a3, a3, value=2).mul_(a3).add_(e)   # 2a³/27 - ab/3 + e (b is not needed after this)
    D = q.square().mul_(0.25).add_(p.div(3).pow_(3))          # (q/2)² + (p/3)³
    three = D <= 0

    # One real root (D > 0): Cardano
    sD = D.clamp_min_(0).sqrt_()
    s1 = q.mul(-0.5)
    s2 = s1 - sD
    s1.add_(sD)
    x1 = s1.sign().mul_(s1.abs_().pow_(1 / 3)).add_(s2.sign().mul_(s2.abs_().pow_(1 / 3)))
    # Three real roots (D ≤ 0, hence p ≤ 0): Viète
    pn = p.clamp_max_(-tiny)
    rho = pn.div(-3).sqrt_().mul_(2)
    phi = (
        q.mul_(1.5).div_(pn)
        .mul_(pn.reciprocal().mul_(-3).sqrt_())
        .clamp_(-1, 1).acos_().div_(3)
    )
    x_hi = phi.cos().mul_(rho)
    x_lo = phi.add_(2 * math.pi / 3).cos_().mul_(rho)

    def polish_(u):
        for _ in range(polish):
            P = (c3 * u).add_(c2).mul_(u).add_(c1).mul_(u).add_(c0)
            dP = (c3 * u).mul_(3).add_(c2, alpha=2).mul_(u).add_(c1)
            u.sub_(P.div_(dP).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0))
        return u

    def log_integrand(u):
        r2 = (M * u).sub_(R0).square_()                       # r²
        s2 = (c * u).add_(V)                                  # σ²
        return (
            u.log().mul_(nu)
            .sub_(u * beta)
            .sub_(r2.add_(dq).div_(s2).mul_(0.5))
            .sub_(s2.log_().mul_(0.5))
        )

    u_hi = polish_(torch.where(three, x_hi, x1).sub_(a3)).clamp_min_(tiny)
    u_lo = polish_(torch.where(three, x_lo, x1).sub_(a3))
    use_lo = three.logical_and_(u_lo > 0)
    u_lo.clamp_min_(tiny)
    use_lo.logical_and_(log_integrand(u_lo) > log_integrand(u_hi))
    U = torch.where(use_lo, u_lo, u_hi)
    # Safety net (should not trigger): fall back to the prior mean
    return torch.where(U.isfinite() & (U > 0), U, nu / beta)


def _maybe_compile(fn):
    """
    Run `fn` through `torch.compile` when that is possible and useful.

    Falls back to the eager function:
    * if compilation is disabled (`_compile(False)`);
    * if we are already inside a compiled region (`torch.compiler.is_compiling()`),
      so that the outer trace simply goes through `fn`;
    * if compiling or running the compiled function raises; a warning is
      emitted once, and the eager function is used from then on.

    The eager function is available as `.eager`.
    """
    state = {"compiled": None, "failed": False}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not COMPILE or state["failed"] or torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        if state["compiled"] is None:
            state["compiled"] = torch.compile(fn)
        try:
            return state["compiled"](*args, **kwargs)
        except Exception as exc:  # noqa: BLE001  (any backend failure)
            state["failed"] = True
            warnings.warn(
                f"torch.compile failed for {fn.__name__}; using eager mode from now on.\n"
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            return fn(*args, **kwargs)

    wrapper.eager = fn
    return wrapper


@_maybe_compile
def _quadrature_kernel(
    Y: Tensor, C: Tensor, V: Tensor, M: Tensor, gain: Tensor, mu: Tensor, nu: Tensor,
    *, dq: float, nodes: tuple[float, ...], logws: tuple[float, ...], full: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Pure-tensor body of `GammaMixtureModel._quadrature` (see there).
    Branch-free and sync-free, so that it can be compiled.

    dq    : quantisation variance δ (every r² below is r² + δ)
    nodes : Gauss-Hermite nodes
    logws : log Gauss-Hermite weights + node² (see `gauss_hermite`)
    full  : noise model ("full" if True, "constant" otherwise)
    """
    beta = nu / mu  # prior rate of the Gamma distribution
    R0 = Y - C  # Residuals at U=0
    M2 = M.square()

    # c    = camera offset (current MAP)
    # v    = background variance (current MAP)
    # m    = modulation (current MAP)
    # γ    = camera gain (current MAP)
    # r₀   = residuals at U=0
    # ν    = prior shape of the Gamma distribution
    # μ    = prior mean of the Gamma distribution
    # β    = prior rate of the Gamma distribution = ν / μ
    # σ²   = variance of the Normal distribution  = v + γ·m·u

    # We want to integrate
    #   I(f) = ∫_{ℝ+} f(u) · p(u,y) du
    # where
    #   p(u,y) = Normal(y | c + m·u, σ²(u)) · Gamma(u | ν, β)
    # and f is some function of u (for the conditional expectations).
    #
    # We re-parameterise the integral in t = log u, so that du = eᵗ dt and
    #   I(f) = ∫_{ℝ} f(eᵗ) · eᵗ · p(eᵗ) dt
    #
    # The integrand is log-concave in t and its mode is found by
    # solving l'(t) = 0, where
    #   l(t) = t + log p(eᵗ)
    # is the log of the integrand. The argument at the mode is t̂ = log û.
    #
    # We can use Gauss-Hermite quadrature to integrate the function,
    # centred on the mode t̂. The integrand is evaluated at the nodes
    # tⱼ = t̂ + √2·s·xⱼ, where xⱼ are the Gauss-Hermite nodes and
    # s² = -1 / l''(t̂) is the curvature at the mode. The integral is
    # calulated as
    #   I(f) ≈ (√2·s) Σⱼ[ wⱼ · uⱼ · p(uⱼ) · f(uⱼ) ]
    # where wⱼ are the Gauss-Hermite weights and uⱼ = exp(tⱼ).

    # --- variance that does not depend on the integration variable
    if full:
        def make_S2(U: Tensor) -> Tensor:
            return V.addcmul(M, U, value=gain)
    else:
        S2 = V.expand_as(Y)

        def make_S2(U: Tensor) -> Tensor:
            return S2

    def make_R(U: Tensor) -> Tensor:
        return R0.addcmul(M, U, value=-1)

    # --- mode of the integrand in t = log u
    if full:
        # σ⁴·l'(t) is a cubic in u (see `_cubic_mode`)
        U = _cubic_mode(R0, V, M, gain, nu, beta, dq)
    else:
        # With a fixed variance, l'(t) = ν - β·u + u·m·(r₀ - m·u)/σ² = 0
        # is a quadratic in u with a single positive root.
        A = M2.div(S2)
        B = (M * R0).div_(S2).sub_(beta).neg_()
        delta = B.hypot((A * nu).mul_(4).sqrt_())
        B0 = (delta - B).div_(A).mul_(0.5)
        B1 = (delta + B).reciprocal_().mul_(nu).mul_(2)  # numerically stable
        U = torch.where(B < 0, B0, B1)
    T = U.log()

    def terms_lognormal(U: Tensor) -> tuple[Tensor, ...]:
        """
        Residual r, variance σ², and the gradient G, Fisher information
        F and exact Hessian H of the Normal (N)LL with respect to u.
        """
        R, S2 = make_R(U), make_S2(U)
        iS2 = S2.reciprocal()
        G = (M * R).mul_(iS2)          # ∂/∂u log N, mean part
        F = M2 * iS2                   # Fisher information, mean part
        if not full:
            return R, S2, iS2, G, F, F
        # Terms that come from the dependence of σ² on u
        tmp1 = (M * iS2).mul_(gain)                                  # γm/σ²
        G.addcmul_(R.square().add_(dq).sub_(S2), tmp1 * iS2, value=0.5)
        F.addcmul_(tmp1, tmp1, value=0.5)                            # + ½γ²m²/σ⁴
        tmp2 = R * tmp1                                              # γmr/σ²
        H = (                                                        # -∂²/∂u² log N
            tmp1.square().mul_(-0.5)
            .addcmul_(M2, iS2)
            .addcmul_(M * tmp2, iS2, value=2)
            .addcmul_(tmp2.square_().addcmul_(tmp1, tmp1, value=dq), iS2)
        )
        return R, S2, iS2, G, F, H

    def log_integrand(T: Tensor, U: Tensor) -> Tensor:
        """l(t) = ν·t - β·u - ½ (r² + δ)/σ² - ½ log σ²   (+ const)"""
        R, S2 = make_R(U), make_S2(U)
        return (
            (nu * T)
            .addcmul_(beta, U, value=-1)
            .addcmul_(R.square_().add_(dq), S2.reciprocal(), value=-0.5)
            .sub_(S2.log(), alpha=0.5)
        )

    # G = LL gradient, F = Fisher information, H = exact NLL Hessian, at the mode
    _, _, _, G, F, H = terms_lognormal(U)

    H0 = beta.sub(G).addcmul_(U, H).mul_(U)
    F0 = U.square().mul_(F).add_(nu)
    msk = (F0 > 1.25 * nu) & (H0 > 0)
    H0 = torch.where(msk, H0, F0)
    scale = H0.reciprocal_().mul_(2).sqrt_()       # √2 · standard deviation
    L0 = log_integrand(T, U)  # log û + log p(û, y) + const, at the mode

    # --- Gauss-Hermite quadrature centred on the mode
    # Z = Σⱼ wⱼ·exp(lⱼ - l₀). The quadrature is centred on the global
    # maximum of l, so lⱼ - l₀ ≤ 0 and the sum cannot overflow: no need for
    # a running log-sum-exp (which costs two transcendentals per node).
    Z = torch.zeros_like(Y)
    EU = torch.zeros_like(Y)        # E[U]      = ∫ u p(u|y) du
    ElogU = torch.zeros_like(Y)     # E[log(u)] = ∫ log(u) p(u|y) du
    Emom = Y.new_zeros((10, *Y.shape))
    for Xj, logWj in zip(nodes, logws):
        Tj = T.add(scale, alpha=Xj)
        Uj = Tj.exp()
        Rj = make_R(Uj)
        Sj = make_S2(Uj)
        iSj = Sj.reciprocal()
        Lj = (
            (nu*Tj)
            .addcmul_(beta, Uj, value=-1)
            .addcmul_(Rj.square().add_(dq), iSj, value=-0.5)
            .add_(iSj.log(), alpha=0.5)
        )
        Wj = Lj.sub_(L0).add_(logWj).exp_()
        Z += Wj
        EU.addcmul_(Wj, Uj)
        ElogU.addcmul_(Wj, Tj)
        Wj *= iSj  # w / σ²
        Emom[_HCC] += Wj
        Emom[_GC].addcmul_(Wj, Rj)
        Emom[_XU].addcmul_(Wj, Uj)
        Emom[_G1].addcmul_(Wj, Uj * Rj)
        Emom[_AU].addcmul_(Wj, Uj.square() if full else Uj.square_())
        Wjj = Wj.mul_(iSj).mul_(0.5)            # w / 2σ⁴
        Wq = Rj.square_().add_(dq).sub_(Sj).mul_(Wjj)  # w · q
        Emom[_HVV] += Wjj
        Emom[_GV] += Wq
        if full:
            Emom[_WU].addcmul_(Wjj, Uj)
            Emom[_G2].addcmul_(Wq, Uj)
            Emom[_BU].addcmul_(Wjj, Uj.square_())

    # log Z = log ∫ p(u,y) du - log û - log p(û, y) - log(√2 / H(û))
    Z = Z.clamp_min_(torch.finfo(Y.dtype).tiny)

    logZGamma = beta.log() * nu - nu.lgamma()          # ν log β - log Γ(ν)
    logZNormal = -0.5 * math.log(2 * math.pi)          # - ½ log 2π
    L1 = (                                             # log ∫ p(u) du
        L0
        .add_(Z.log())
        .add_(scale.log_())
        .add_(logZGamma + logZNormal)
    )

    EU /= Z
    ElogU /= Z
    Emom /= Z

    return L1, EU, ElogU, Emom

