"""Tests of optix.special.bessel_j0, bessel_j1 and bessel_j012."""

import math
from fractions import Fraction

import pytest
import torch

from optix.special import bessel_j0, bessel_j012, bessel_j1

F64 = torch.float64


def _series(order, x, terms=120):
    """Power series of Jₙ summed in exact rational arithmetic."""
    xf = Fraction(x)
    total, term = Fraction(0), (xf / 2) ** order / math.factorial(order)
    for m in range(terms):
        total += term
        term *= -((xf / 2) ** 2) / ((m + 1) * (m + 1 + order))
    return float(total)


def test_bessel_accurate_in_float64():
    xs = [0.0, 0.3, 2.404825557695773, 5.5, 7.0, 12.3, 19.9, 24.9, -6.1]
    x = torch.tensor(xs, dtype=F64)
    for n, fn in ((0, bessel_j0), (1, bessel_j1)):
        ref = torch.tensor([_series(n, v) for v in xs], dtype=F64)
        assert (fn(x) - ref).abs().max() < 2e-15
    ref2 = torch.tensor([_series(2, v) for v in xs], dtype=F64)
    assert (bessel_j012(x)[2] - ref2).abs().max() < 2e-15
    # continuity with torch above the switch point; float32 unchanged
    y = torch.tensor([25.0, 40.0, 300.0], dtype=F64)
    assert torch.equal(bessel_j0(y), torch.special.bessel_j0(y))
    z = torch.linspace(0, 30, 31)
    assert torch.equal(bessel_j1(z), torch.special.bessel_j1(z))


def test_bessel_j012_against_mpmath():
    mpmath = pytest.importorskip("mpmath")
    x = torch.cat(
        [
            torch.linspace(0, 30, 301, dtype=F64),
            torch.tensor([1e-8, 4.9, 5.0, 5.1, 24.99, 25.0, 80.0, 1e3]),
            -torch.tensor([0.3, 7.0, 40.0], dtype=F64),
        ]
    )
    out = bessel_j012(x)
    for n, j in enumerate(out):
        ref = torch.tensor(
            [float(mpmath.besselj(n, float(v))) for v in x], dtype=F64
        )
        assert (j - ref).abs().max() < 5e-15


def test_bessel_j012_float32(device):
    x = torch.linspace(0, 60, 601, dtype=F64)
    ref = bessel_j012(x)
    out = bessel_j012(x.float().to(device))
    for a, b in zip(out, ref, strict=True):
        assert a.dtype == torch.float32
        assert (a.cpu().double() - b).abs().max() < 2e-6


def test_bessel_j012_matches_j0_j1():
    x = torch.linspace(-40, 40, 4001, dtype=F64)
    j0, j1, j2 = bessel_j012(x)
    assert torch.equal(j0, bessel_j0(x))
    assert torch.equal(j1, bessel_j1(x))
    # recurrence J₂ = 2 J₁ / x − J₀ away from 0
    far = x.abs() > 1
    rec = 2 * j1[far] / x[far] - j0[far]
    assert (j2[far] - rec).abs().max() < 1e-14


def test_bessel_small_argument_relative_accuracy():
    x = torch.tensor([1e-100, 1e-12, 1e-9, -1e-8, 3e-5, 1e-3], dtype=F64)
    j0, j1, j2 = bessel_j012(x)
    for n, j in ((0, j0), (1, j1), (2, j2)):
        ref = torch.tensor([_series(n, v) for v in x.tolist()], dtype=F64)
        assert ((j - ref) / ref).abs().max() < 1e-15
    assert float(bessel_j1(torch.tensor(1e-12, dtype=F64))) == 5e-13


def test_bessel_shapes_blocks_and_dtypes():
    # more elements than one block of the rule, any shape
    x = torch.linspace(0, 24, 7 * 14100, dtype=F64).reshape(-1, 7)
    j0 = bessel_j0(x)
    assert j0.shape == x.shape
    torch.testing.assert_close(
        j0, torch.special.bessel_j0(x), rtol=0, atol=1e-6
    )
    # integer input is cast to the default dtype
    out = bessel_j1(torch.tensor([0, 1, 2]))
    assert out.dtype == torch.get_default_dtype()
    assert bessel_j012(torch.zeros((), dtype=F64))[2].shape == ()
