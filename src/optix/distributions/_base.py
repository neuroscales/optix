"""Distribution base class: parameterisations, shapes, dispatch."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, NamedTuple, Self

import torch
from torch import Tensor

from .._config import resolve_backend

if TYPE_CHECKING:
    from ._stats import SuffStats

__all__ = ["Distribution", "kl_divergence", "posterior"]

Form = tuple[str, ...]


def _is_compiling() -> bool:
    fn = getattr(torch.compiler, "is_compiling", None)
    return bool(fn()) if fn is not None else False


def _fmt(shape: Sequence[int]) -> str:
    """`(8, 3)` / `(4,)` / `()`."""
    shape = tuple(shape)
    return str(shape)


def _normalise(
    name: str,
    values: Sequence[object],
    names: Sequence[str],
    event_ndims: Sequence[int],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[tuple[Tensor, ...], torch.Size, tuple[torch.Size, ...]]:
    """Convert parameters to tensors on one dtype and device.

    Returns `(raw, batch_shape, event_shapes)`. Tensors are not copied
    unless their dtype or device changes.
    """
    tensors = [v for v in values if isinstance(v, Tensor)]
    if dtype is None:
        floats = [t.dtype for t in tensors if t.is_floating_point()]
        if floats:
            dtype = floats[0]
            for d in floats[1:]:
                dtype = torch.promote_types(dtype, d)
    if device is None:
        devices = {}
        for n, v in zip(names, values, strict=True):
            if isinstance(v, Tensor):
                devices.setdefault(v.device, n)
        if len(devices) > 1:
            parts = ", ".join(f"{n} on {d}" for d, n in devices.items())
            raise ValueError(
                f"{name}: parameters are on different devices: {parts}"
            )
        if devices:
            device = next(iter(devices))
    if dtype is None or device is None:
        backend = resolve_backend(dtype=dtype, device=device)
        dtype = dtype or backend["dtype"]
        device = device or backend["device"]
    raw = tuple(
        torch.as_tensor(v).to(dtype=dtype, device=device) for v in values
    )
    batch, events = _shapes(name, names, raw, event_ndims)
    return raw, batch, events


def _shapes(
    name: str,
    names: Sequence[str],
    raw: Sequence[Tensor],
    event_ndims: Sequence[int],
) -> tuple[torch.Size, tuple[torch.Size, ...]]:
    """Broadcast batch shape and per-parameter event shapes."""
    batches, events = [], []
    for n, t, e in zip(names, raw, event_ndims, strict=True):
        if t.ndim < e:
            raise ValueError(
                f"{name}: {n} needs at least {e} dim"
                f"{'s' if e > 1 else ''}, got shape {_fmt(t.shape)}"
            )
        batches.append(t.shape[: t.ndim - e])
        events.append(t.shape[t.ndim - e :])
    try:
        batch = torch.broadcast_shapes(*batches)
    except RuntimeError:
        parts = ", ".join(
            f"{n} {_fmt(b)}" for n, b in zip(names, batches, strict=True)
        )
        raise ValueError(
            f"{name}: parameter batch shapes are not broadcastable: {parts}"
        ) from None
    return batch, tuple(events)


def _check_value(name: str, param: str, kind: str, value: Tensor) -> None:
    """Raise if `value` violates the constraint `kind` (synchronises)."""
    if kind == "real":
        ok, what = torch.isfinite(value).all(), "finite"
    elif kind == "positive":
        ok = (torch.isfinite(value) & (value > 0)).all()
        what = "positive and finite"
    elif kind == "nonnegative":
        ok = (torch.isfinite(value) & (value >= 0)).all()
        ok = ok & (value.sum(-1) > 0).all()
        what = "non-negative and finite, with a positive sum"
    elif kind == "logits":
        ok = (~torch.isnan(value) & (value < torch.inf)).all()
        ok = ok & torch.isfinite(torch.logsumexp(value, -1)).all()
        what = "below +inf, not NaN, and not all -inf"
    else:  # pragma: no cover - programming error
        raise AssertionError(kind)
    if not bool(ok):
        raise ValueError(f"{name}: {param} must be {what}")


class Distribution:
    """Base of all distributions: parameterisations, shapes and dispatch.

    **Parameterisations.** A family declares several parameterisations
    (`_forms`: tuples of parameter names). An object stores exactly the
    tensors of the one it was built with
    ([`form`][optix.distributions.Distribution.form]) and derives every
    other parameter on demand (`_derive`), e.g. the precision of a
    Normal given by its variance. `params()` returns the canonical
    parameterisation (`_canonical`), which is what the
    [`functional`][optix.distributions.functional] kernels take.

    Later families declare their forms the same way, e.g. (sketch):

    ```python
    class MultivariateNormal(Distribution):
        _forms = {
            ("mean", "covariance"): (1, 2),
            ("mean", "precision"): (1, 2),
            ("mean", "scale_tril"): (1, 2),
            ("mean", "precision_tril"): (1, 2),
        }
        _canonical = ("mean", "precision_tril")
        _event_ndim = 1
        # _check_events: mean (K,) against matrices (K, K)
        # _derive("precision_tril"): cholesky / flip trick, ...


    class Wishart(Distribution):
        _forms = {("df", "scale"): (0, 2), ("df", "rate"): (0, 2),
                  ("df", "mean"): (0, 2)}   # mean = df * scale
    ```

    and a conjugate update returns the form in which it is additive (a
    Wishart posterior in `(df, rate)`).

    **Caching.** Derived quantities are recomputed on every access unless
    caching is switched on (`cache=True` at construction, or
    `q.cache = True`). An entry is reused only while the stored tensors
    keep the version counters they had when it was computed, so an
    in-place change of a parameter (`s.mul_(2)`) invalidates it. The
    cache is bypassed while compiling (traced functions have no side
    effects and never see it), when a stored tensor requires gradients (a
    cached graph would break a second backward pass) and for inference
    tensors (which have no version counter). A derived parameter has the
    broadcast shape of the stored tensors it is computed from (e.g. the
    rate of a Gamma given by a scalar shape and a per-pixel mean is
    per-pixel), and is cached with that shape.

    **Shapes.** Parameters are stored with their own shapes (not expanded
    to the batch shape), so a scalar prior against per-pixel posteriors
    costs nothing; public parameter attributes and method results are
    `expand` views of shape `batch_shape + event`. **Returned tensors
    must not be modified in place** (`clone()` first).

    **dtype and device** follow the parameter tensors (promoted between
    floating dtypes); the configuration is used only when no parameter is
    a floating-point tensor.

    Objects are immutable by convention. Values are not checked unless
    `validate=True` is passed or
    [`validate`][optix.distributions.Distribution.validate] is called (both
    synchronise with the host).
    """

    # No `__slots__`: torch 2.3's dynamo reads the attributes of an
    # instance of a slotted class with a plain `getattr` instead of
    # inlining properties, which evaluates them eagerly and installs
    # guards on short-lived tensors (`InternalTorchDynamoError` when an
    # object is passed to a compiled function).
    _form: Form
    _stored: tuple[Tensor, ...]
    _batch_shape: torch.Size
    _event_shape: torch.Size
    _cache: dict[str, tuple[tuple[int, ...], Tensor]] | None
    #: Whether value checks that synchronise are run (`validate=True`).
    _validate: bool

    #: Parameterisations: names -> event ndim of each parameter.
    _forms: ClassVar[dict[Form, tuple[int, ...]]] = {}
    #: Parameterisation taken by the `functional` kernels.
    _canonical: ClassVar[Form] = ()
    #: Constraint of each parameter name (`real`, `positive`, ...).
    _constraints: ClassVar[dict[str, str]] = {}
    #: Event ndim of every parameter name, stored or derived.
    _ndims: ClassVar[dict[str, int]] = {}
    #: Number of event dims of the distribution.
    _event_ndim: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        ndims = dict(cls._ndims)
        for names, nd in cls._forms.items():
            ndims.update(zip(names, nd, strict=True))
        cls._ndims = ndims

    # -- construction ----------------------------------------------------
    def _init(
        self,
        form: Form,
        values: Sequence[object],
        *,
        validate: bool,
        cache: bool,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> None:
        name = type(self).__name__
        raw, _, _ = _normalise(
            name, values, form, self._forms[form], dtype, device
        )
        self._setup(form, raw)
        self.cache = cache
        self._validate = bool(validate)
        if validate:
            self.validate()

    def _setup(self, form: Form, raw: Sequence[Tensor]) -> None:
        name = type(self).__name__
        batch, events = _shapes(name, form, raw, self._forms[form])
        self._form = form
        self._raw = raw
        self._batch_shape = batch
        self._event_shape = self._check_events(form, events)
        self._cache = None
        self._validate = False

    @classmethod
    def _check_events(
        cls, form: Form, events: Sequence[torch.Size]
    ) -> torch.Size:
        """Check parameter event shapes; return the distribution's.

        Default: every parameter with event dims has the same event shape
        (no broadcasting), and the distribution's event shape is its last
        `_event_ndim` dims. Families whose parameters have different
        event ranks (a mean `(K,)` and a precision `(K, K)`) override
        this.
        """
        shaped = [(n, e) for n, e in zip(form, events, strict=True) if e]
        if any(e != shaped[0][1] for _, e in shaped):
            parts = ", ".join(f"{n} {_fmt(e)}" for n, e in shaped)
            raise ValueError(f"{cls.__name__}: event shapes differ: {parts}")
        if not shaped:
            return torch.Size(())
        event = shaped[0][1]
        return event[len(event) - cls._event_ndim :]

    @classmethod
    def _new(cls, *raw: Tensor, form: Form | None = None) -> Self:
        """Build from tensors in `form` (default canonical).

        Shapes are checked; values are never validated. Floating tensors
        are promoted to a common dtype (a no-op when they already agree).
        Used by library code (updates, fits, `expand`, ...).
        """
        floats = [t.dtype for t in raw if t.is_floating_point()]
        if floats:
            dtype = floats[0]
            for d in floats[1:]:
                dtype = torch.promote_types(dtype, d)
            raw = tuple(
                t.to(dtype) if t.is_floating_point() else t for t in raw
            )
        self = object.__new__(cls)
        self._setup(cls._canonical if form is None else form, raw)
        return self

    @staticmethod
    def _pick(
        name: str, given: dict[str, object], forms: Sequence[Form]
    ) -> Form:
        """The form whose parameters are exactly the non-None `given`."""
        names = {k for k, v in given.items() if v is not None}
        for form in forms:
            if set(form) == names:
                return form
        options = " or ".join("(" + ", ".join(f) + ")" for f in forms)
        got = ", ".join(sorted(names)) or "nothing"
        raise ValueError(f"{name}: give {options}; got {got}")

    # -- parameters ------------------------------------------------------
    @property
    def _raw(self) -> tuple[Tensor, ...]:
        """Stored tensors, in `form` order."""
        return self._stored

    @_raw.setter
    def _raw(self, value: Sequence[Tensor]) -> None:
        # every path that replaces a stored tensor goes through here
        self._stored = tuple(value)
        if getattr(self, "_cache", None) is not None:
            self._cache = {}

    @property
    def form(self) -> Form:
        """Names of the stored parameters (the parameterisation given)."""
        return self._form

    @property
    def cache(self) -> bool:
        """Whether derived quantities are cached (off while compiling)."""
        return self._cache is not None

    @cache.setter
    def cache(self, value: bool) -> None:
        if not value:
            self._cache = None
        elif self._cache is None:
            self._cache = {}

    def _derive(self, name: str) -> Tensor:
        """Compute parameter `name` from the stored ones (unexpanded)."""
        raise AttributeError(
            f"{type(self).__name__} cannot derive {name!r} from {self._form}"
        )

    def _get(self, name: str) -> Tensor:
        """Parameter `name`, stored or derived, with its own shape."""
        if name in self._form:
            return self._raw[self._form.index(name)]
        # test compiling first: traced code never reads the cache
        if _is_compiling() or self._cache is None:
            return self._derive(name)
        raw = self._raw
        if any(t.requires_grad or t.is_inference() for t in raw):
            return self._derive(name)
        key = tuple(t._version for t in raw)
        hit = self._cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        value = self._derive(name)
        self._cache[name] = (key, value)
        return value

    def _view(self, name: str) -> Tensor:
        """Parameter `name` expanded to `batch_shape + its event`."""
        value = self._get(name)
        e = self._ndims[name]
        return value.expand(self._batch_shape + value.shape[value.ndim - e :])

    def _out(self, value: Tensor, event_ndim: int = 0) -> Tensor:
        """Expand a method result to `batch_shape + its event`."""
        event = value.shape[value.ndim - event_ndim :]
        return value.expand(self._batch_shape + event)

    def params(self) -> tuple[Tensor, ...]:
        """Canonical parameters, unexpanded, for `functional` kernels."""
        return tuple(self._get(n) for n in self._canonical)

    def stored(self) -> tuple[Tensor, ...]:
        """Stored parameters, unexpanded, in
        [`form`][optix.distributions.Distribution.form] order."""
        return self._raw

    def validate(self) -> Self:
        """Check parameter values (synchronises); return `self`.

        Skipped while compiling.

        Raises
        ------
        ValueError
            If a stored parameter violates its constraint.
        """
        if _is_compiling():
            return self
        name = type(self).__name__
        for n, t in zip(self._form, self._raw, strict=True):
            _check_value(name, n, self._constraints.get(n, "real"), t)
        return self

    # -- shapes ----------------------------------------------------------
    @property
    def batch_shape(self) -> torch.Size:
        """Broadcast batch shape of the parameters."""
        return self._batch_shape

    @property
    def event_shape(self) -> torch.Size:
        """Shape of one sample."""
        return self._event_shape

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype of the parameters."""
        return self._raw[0].dtype

    @property
    def device(self) -> torch.device:
        """Device of the parameters."""
        return self._raw[0].device

    def _rebuild(self, raw: Sequence[Tensor]) -> Self:
        new = self._new(*raw, form=self._form)
        new.cache = self.cache
        new._validate = self._validate
        return new

    def expand(self, batch_shape: Sequence[int]) -> Self:
        """Same distribution with a larger batch shape (no copy).

        Parameters
        ----------
        batch_shape : sequence of int
            Target batch shape, broadcast-compatible with
            [`batch_shape`][optix.distributions.Distribution.batch_shape].

        Returns
        -------
        Distribution
            A new object sharing the stored tensors.
        """
        shape = torch.Size(batch_shape)
        try:
            ok = torch.broadcast_shapes(self._batch_shape, shape) == shape
        except RuntimeError:
            ok = False
        if not ok:
            raise ValueError(
                f"{type(self).__name__}.expand: batch shape "
                f"{_fmt(self._batch_shape)} cannot expand to {_fmt(shape)}"
            )
        new = self._rebuild(self._raw)
        new._batch_shape = shape
        return new

    def __getitem__(self, index: object) -> Self:
        """Index the batch dims (basic indexing only); event dims kept."""
        index = index if isinstance(index, tuple) else (index,)
        for i in index:
            ok = i is None or i is Ellipsis or isinstance(i, int | slice)
            if not ok or isinstance(i, bool):
                raise TypeError(
                    f"{type(self).__name__}: only int, slice, None and "
                    f"Ellipsis indices are supported, got {type(i).__name__}"
                )
        used = sum(i is not None and i is not Ellipsis for i in index)
        if used > len(self._batch_shape):
            raise IndexError(
                f"{type(self).__name__}: too many indices ({used}) for "
                f"batch shape {_fmt(self._batch_shape)}"
            )
        ellipsis = any(i is Ellipsis for i in index)
        raw = []
        for n, t in zip(self._form, self._raw, strict=True):
            e = self._ndims[n]
            full = t.expand(self._batch_shape + t.shape[t.ndim - e :])
            # after an Ellipsis, full slices keep the event dims in place
            v = full[index + (slice(None),) * e if ellipsis else index]
            # broadcast batch dims (stride 0) go back to size 1, so that
            # derived parameters are not computed at the full batch shape
            for d in range(v.ndim - e):
                if v.stride(d) == 0 and v.shape[d] > 1:
                    v = v.narrow(d, 0, 1)
            raw.append(v)
        batch = torch.empty(self._batch_shape, device="meta")[index].shape
        new = self._rebuild(raw)
        new._batch_shape = batch
        return new

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Copy on another device and/or floating dtype (new object)."""
        raw = []
        for t in self._raw:
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            raw.append(t)
        return self._rebuild(raw)

    def detach(self) -> Self:
        """Same distribution with detached parameters."""
        return self._rebuild([t.detach() for t in self._raw])

    def __repr__(self) -> str:
        dtype = str(self.dtype).removeprefix("torch.")
        return (
            f"{type(self).__name__}({', '.join(self._form)}; "
            f"batch_shape={_fmt(self._batch_shape)}, "
            f"event_shape={_fmt(self._event_shape)}, "
            f"dtype={dtype}, device={self.device})"
        )

    # -- maths (subclasses) ----------------------------------------------
    def _unsupported(self, what: str) -> NotImplementedError:
        return NotImplementedError(
            f"{type(self).__name__} does not implement {what}"
        )

    @property
    def mean(self) -> Tensor:
        """`E[x]`."""
        raise self._unsupported("mean")

    @property
    def variance(self) -> Tensor:
        """Var[x] (elementwise / marginal)."""
        raise self._unsupported("variance")

    def entropy(self) -> Tensor:
        """Differential (or discrete) entropy, shape `batch_shape`."""
        raise self._unsupported("entropy")

    def log_prob(self, value: Tensor) -> Tensor:
        """Log-density at `value` of shape `(*sample, *batch, *event)`."""
        raise self._unsupported("log_prob")

    def natural_params(self) -> tuple[Tensor, ...]:
        """Natural parameters, ordered like
        [`expected_stats`][optix.distributions.Distribution.expected_stats]."""
        raise self._unsupported("natural_params")

    def log_normalizer(self) -> Tensor:
        """Log-normalizer A(η), shape `batch_shape`."""
        raise self._unsupported("log_normalizer")

    def expected_stats(self) -> NamedTuple:
        """E[T(x)], ordered like
        [`natural_params`][optix.distributions.Distribution.natural_params]."""
        raise self._unsupported("expected_stats")

    @classmethod
    def from_natural(cls, *eta: Tensor) -> Self:
        """Distribution with natural parameters `eta` (η)."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_natural"
        )

    def to_torch(self) -> torch.distributions.Distribution:
        """Equivalent `torch.distributions` object (no validation)."""
        raise self._unsupported("to_torch")

    def _kl_event_shape(self) -> torch.Size:
        """Shape that must agree between the operands of a KL.

        The event shape, except for families with several views of the
        same variable (`Categorical` labels / one-hot).
        """
        return self._event_shape

    def _kl_event_desc(self) -> str:
        """What `_kl_event_shape` is, for error messages."""
        return f"event_shape {_fmt(self._event_shape)}"

    def _check_value_shape(self, value: Tensor) -> torch.Size:
        """Check the event dims of `value`; return the result shape."""
        e = len(self._event_shape)
        event = value.shape[value.ndim - e :] if e else torch.Size(())
        if value.ndim < e or event != self._event_shape:
            raise ValueError(
                f"{type(self).__name__}.log_prob: value of shape "
                f"{_fmt(value.shape)} does not end with the event shape "
                f"{_fmt(self._event_shape)}"
            )
        try:
            return torch.broadcast_shapes(
                value.shape[: value.ndim - e], self._batch_shape
            )
        except RuntimeError:
            raise ValueError(
                f"{type(self).__name__}.log_prob: value batch shape "
                f"{_fmt(value.shape[: value.ndim - e])} is not "
                f"broadcastable to {_fmt(self._batch_shape)}"
            ) from None

    # -- sugar -----------------------------------------------------------
    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        """Samples (no gradient), delegated to
        [`to_torch`][optix.distributions.Distribution.to_torch]."""
        return self.to_torch().sample(torch.Size(sample_shape))

    def rsample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        """Reparameterised samples, delegated to
        [`to_torch`][optix.distributions.Distribution.to_torch]."""
        return self.to_torch().rsample(torch.Size(sample_shape))

    def kl(self, other: "Distribution") -> Tensor:
        """`kl_divergence(self, other)`."""
        return kl_divergence(self, other)

    def posterior(self, stats: "SuffStats", **known: Tensor) -> Self:
        """`posterior(self, stats, **known)`."""
        return posterior(self, stats, **known)


# -- registries ----------------------------------------------------------

_KL: dict[tuple[type, type], Callable[..., Tensor]] = {}
_Update = Callable[..., Distribution]
_POSTERIOR: dict[tuple[type, type], _Update] = {}


def _lookup(
    registry: dict[tuple[type, type], Callable], a: type, b: type
) -> Callable | None:
    for x in a.__mro__:
        for y in b.__mro__:
            fn = registry.get((x, y))
            if fn is not None:
                return fn
    return None


def _register_kl(
    type_q: type, type_p: type
) -> Callable[[Callable[..., Tensor]], Callable[..., Tensor]]:
    def decorate(fn: Callable[..., Tensor]) -> Callable[..., Tensor]:
        _KL[(type_q, type_p)] = fn
        return fn

    return decorate


def _register_posterior(
    type_prior: type, type_stats: type
) -> Callable[[_Update], _Update]:
    def decorate(fn: _Update) -> _Update:
        _POSTERIOR[(type_prior, type_stats)] = fn
        return fn

    return decorate


def kl_divergence(q: Distribution, p: Distribution) -> Tensor:
    """KL(q ‖ p), elementwise over the broadcast batch shape.

    Parameters
    ----------
    q, p : Distribution
        Distributions with the same event shape and broadcastable batch
        shapes. Broadcasting is right-aligned: a prior with one value per
        class `(nc,)` matches a posterior `(*batch, nc)`; with the
        class axis first, `(nc, H, W)`, index the prior as
        `p[:, None, None]`.

    Returns
    -------
    Tensor
        Shape `broadcast(q.batch_shape, p.batch_shape)`; event dims are
        reduced, batch dims never are. Neither operand is expanded: the
        kernels run on each operand's own parameter shapes and only the
        final arithmetic broadcasts. The value is not clamped at zero.

    Raises
    ------
    NotImplementedError
        If the pair of types has no registered divergence.
    ValueError
        If the event shapes differ, the batch shapes do not broadcast or
        the devices differ.
    """
    fn = _lookup(_KL, type(q), type(p))
    nq, np_ = type(q).__name__, type(p).__name__
    if fn is None:
        raise NotImplementedError(f"KL({nq} || {np_}) is not implemented")
    if q._kl_event_shape() != p._kl_event_shape():
        raise ValueError(
            f"kl_divergence: event shapes differ: q is {nq} with "
            f"{q._kl_event_desc()}, p is {np_} with {p._kl_event_desc()}"
        )
    try:
        batch = torch.broadcast_shapes(q.batch_shape, p.batch_shape)
    except RuntimeError:
        raise ValueError(
            "kl_divergence: batch shapes are not broadcastable: q "
            f"{_fmt(q.batch_shape)}, p {_fmt(p.batch_shape)}"
        ) from None
    if q.device != p.device:
        raise ValueError(
            f"kl_divergence: q is on {q.device}, p is on {p.device}"
        )
    return fn(q, p).expand(batch)


def posterior(
    prior: Distribution, stats: "SuffStats", **known: Tensor
) -> Distribution:
    """Conjugate posterior of `prior` given likelihood statistics.

    Parameters
    ----------
    prior : Distribution
        Conjugate prior.
    stats : SuffStats
        Weighted sufficient statistics of the observations (weights must
        be non-negative; this is not checked). Tempering is
        `posterior(prior, stats.scale(c))`.
    **known : Tensor
        Fixed parameters of the likelihood, required by some pairs (e.g.
        `shape` for a Gamma likelihood, `mean` or `precision` for a
        Normal one).

    Returns
    -------
    Distribution
        Same type as `prior`, batch shape
        `broadcast(prior.batch_shape, stats.batch_shape, known)`, stored
        in the parameterisation where the update is additive. No value is
        validated and nothing synchronises.

    Raises
    ------
    NotImplementedError
        If the pair has no registered update.
    ValueError
        If the event shapes differ or the batch shapes do not broadcast.
    TypeError
        If a required `known` parameter is missing.
    """
    fn = _lookup(_POSTERIOR, type(prior), type(stats))
    name, sname = type(prior).__name__, type(stats).__name__
    if fn is None:
        raise NotImplementedError(
            f"posterior: no conjugate update of a {name} prior from {sname}"
        )
    if prior.event_shape != stats.event_shape:
        raise ValueError(
            f"posterior: event shapes differ: prior {name} "
            f"{_fmt(prior.event_shape)}, {sname} {_fmt(stats.event_shape)}"
        )
    try:
        torch.broadcast_shapes(prior.batch_shape, stats.batch_shape)
    except RuntimeError:
        raise ValueError(
            f"posterior: batch shapes are not broadcastable: prior {name} "
            f"{_fmt(prior.batch_shape)}, {sname} {_fmt(stats.batch_shape)}"
        ) from None
    return fn(prior, stats, **known)
