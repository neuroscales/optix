"""Generate `destripe_fixtures.npz` (run in a separate venv).

References: PyWavelets 1.8.0 (MIT licence) and algotom 1.7.0 (Apache-2.0
licence), numpy, float64:

```text
uv venv -p 3.12 ref-venv
VIRTUAL_ENV=ref-venv uv pip install pywavelets==1.8.0 algotom==1.7.0
ref-venv/bin/python tests/data/make_destripe_fixtures.py
```

Stored arrays
-------------
image : (61, 70) float64
    Smooth content plus vertical stripes (numpy only, no optix import).
db<N>_rec_lo : (2N,) float64
    PyWavelets' `Wavelet("dbN").rec_lo`, N = 1, ..., 38.
dwt_<wavelet>_<i> : float64
    `pywt.wavedec2(image, wavelet, mode="symmetric", level=3)`,
    flattened in order (`a3, h3, v3, d3, ..., h1, v1, d1`), for
    `db1`, `db2` and `db7`.
fig5_<case> : (61, 70) float64
    The Matlab code of Münch et al. 2009, Fig. 5, transcribed with
    PyWavelets (`dwt2` / `idwt2` in `"symmetric"` mode, which is Matlab's
    `"sym"` mode, `numpy.fft` for `fft` / `fftshift`), output cropped to
    the input shape. Cases (wavelet, L, σ): `a` = (db4, 3, 1.5) on
    `image`, `b` = (db10, 2, 4.0) on `image`, `c` = (db4, 3, 1.5) on
    `image.T` then transposed back (horizontal stripes of `image.T`).
algotom_image : (71, 71) float64
    Input of the algotom case.
algotom : (71, 71) float64
    `algotom.prep.removal.remove_stripe_based_wavelet_fft(algotom_image,
    level=2, size=1.5, wavelet_name="db3")`. With this size, every
    padded vertical-detail band has an odd number of rows, where
    algotom's damping window is centred on the zero frequency.
algotom_pad : int
    The padding algotom applies (mean along rows, edge along columns).
"""

import os

import algotom.prep.removal as removal
import numpy as np
import pywt

HERE = os.path.dirname(os.path.abspath(__file__))


def make_image(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]] / max(shape)
    content = np.sin(6 * x + 2 * y) + np.exp(
        -((x - 0.5) ** 2 + (y - 0.4) ** 2) * 20
    )
    stripes = rng.normal(0, 0.3, shape[1])[None, :]
    return content + stripes + 0.01 * rng.normal(size=shape)


def fig5(ima: np.ndarray, dec_num: int, wname: str, sigma: float):
    """Fig. 5 of Münch et al. 2009, line by line."""
    shape = ima.shape
    ch, cv, cd = [], [], []
    for _ in range(dec_num):
        ima, (h, v, d) = pywt.dwt2(ima, wname, mode="symmetric")
        ch.append(h)
        cv.append(v)
        cd.append(d)
    for ii in range(dec_num):
        fcv = np.fft.fftshift(np.fft.fft(cv[ii], axis=0), axes=0)
        my, mx = fcv.shape
        k = np.arange(-(my // 2), -(my // 2) + my)
        damp = 1 - np.exp(-(k**2) / (2 * sigma**2))
        fcv = fcv * damp[:, None]
        cv[ii] = np.real(np.fft.ifft(np.fft.ifftshift(fcv, axes=0), axis=0))
    nima = ima
    for ii in range(dec_num - 1, -1, -1):
        nima = nima[: ch[ii].shape[0], : ch[ii].shape[1]]
        nima = pywt.idwt2(
            (nima, (ch[ii], cv[ii], cd[ii])), wname, mode="symmetric"
        )
    return nima[: shape[0], : shape[1]]


def main() -> None:
    image = make_image((61, 70))
    out = {"image": image}
    for n in range(1, 39):
        out[f"db{n}_rec_lo"] = np.asarray(pywt.Wavelet(f"db{n}").rec_lo)
    for name in ("db1", "db2", "db7"):
        coeffs = pywt.wavedec2(image, name, mode="symmetric", level=3)
        flat = [coeffs[0]] + [c for det in coeffs[1:] for c in det]
        for i, c in enumerate(flat):
            out[f"dwt_{name}_{i}"] = c
    out["fig5_a"] = fig5(image, 3, "db4", 1.5)
    out["fig5_b"] = fig5(image, 2, "db10", 4.0)
    out["fig5_c"] = fig5(image.T.copy(), 3, "db4", 1.5).T
    sino = make_image((71, 71), seed=1)
    pad = min(150, int(0.1 * min(sino.shape)))
    padded = np.pad(sino, pad)
    coeffs = pywt.wavedec2(padded, "db3", level=2)
    assert all(det[1].shape[0] % 2 == 1 for det in coeffs[1:])
    out["algotom_image"] = sino
    out["algotom"] = removal.remove_stripe_based_wavelet_fft(
        sino, level=2, size=1.5, wavelet_name="db3"
    )
    out["algotom_pad"] = np.asarray(pad)
    np.savez_compressed(os.path.join(HERE, "destripe_fixtures.npz"), **out)


if __name__ == "__main__":
    main()
