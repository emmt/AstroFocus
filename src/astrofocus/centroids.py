# -*- coding: utf-8 -*-

"""Implement centroiding methods such as the center of gravity."""

import copy
import time
import numpy as np

def fix_weights(w):
    """Set weights to zero where negative or NaN."""
    w = np.maximum(w, 0)
    cond = np.isnan(w)
    if cond.any():
        w[cond] = 0
    return w

def center_of_gravity(w, x, y, *, covariance=False, gamma=None, sigma=None):
    """Compute the 2-dimensional center of gravity and, optionally, its
    covariance.

    Argument `w` specifies the weights which must all be finite and
    non-negative (this is not checked). If this is not the case, you may call
    `centroids.fix_weights(w)`.

    Arguments `x` and `y` specify the coordinates at each weight (they must all
    be finite, this is not checked).

    The shapes of arguments `w`, `x`, and `y` must be conformable.

    Keyword `covariance` specifies whether to also return the covariance of the
    center of gravity. If `covariance` is `False`, the result is the 2-tuple
    ``(xc, yc)`` with the coordinates of the center of gravity. If `covariance`
    is `True`, the result is the 5-tuple ``(xc, yc, Cxx, Cxy, Cyy)`` where
    `Cxx`, `Cxy`, and `Cyy` are the coefficients of the center of gravity
    covariance. Three different noise models are possible:

    - If `sigma` is `None` or `0`, it is assumed that independent Poisson noise
      dominates with a variance given by `w` if `gamma` is `None` (the default)
      or by `gamma*w` otherwise.

    - Otherwise, if `gamma` is `None` or `0`, i.i.d. noise of variance
      `sigma**2` is assumed.

    - Otherwise, a mixture of independent Poisson plus Gaussian noises is
      assumed with variance given by `gamma*w + sigma**2`.

    Keyword `gamma` is to specify the detector gain in ADU per electron. This
    keyword is ignored if `covariance` is `False`.

    Keyword `signa` is to specify the standard deviation in ADU of the signal
    independent noise (i.e. not the photon noise). This keyword is ignored if
    `covariance` is `False`.

    You may call ``centroids.invert_2x2_symmetric_matrix(Cxx, Cxy, Cyy)`` to
    compute the precision matrix of the center of gravity.

    """
    assert isinstance(covariance, bool)
    # Make sure that `w`, `x`, and `y` are conformable `np.ndarrays` with the
    # same shape. For small regions of interest, calling `np.broadcast_arrays`
    # has non-negligible overheads, so we first check whether this is necessary
    # (which is very fast).
    if not (isinstance(w, np.ndarray) and
            isinstance(x, np.ndarray) and
            isinstance(x, np.ndarray)) or not w.shape == x.shape == y.shape:
        w, x, y = np.broadcast_arrays(w, x, y) # also converts to `np.ndarray`

    # Compute center of gravity.
    sw = w.sum() # sum of weights
    xc = (w*x).sum()/sw
    yc = (w*y).sum()/sw
    if not covariance:
        return xc, yc

    # Compute the covariance for the noise model selected by `gamma` and `sigma`.
    dx = x - xc
    dy = y - yc
    if sigma is None or sigma == 0:
        # Assume that Poisson noise dominates.
        if gamma is None:
            gamma = 1
        else:
            assert gamma > 0
        q = w
        r = gamma/sw**2
    elif gamma is None or gamma == 0:
        # Assume i.i.d. noise of variance sigma**2.
        assert sigma > 0
        q = None
        r = (sigma/sw)**2
    else:
        # Assume a mixture model of noise.
        assert gamma > 0
        assert sigma > 0
        q = gamma*w + sigma**2
        r = (1/sw)**2

    if q is None:
        # No signal-dependent variance.
        Cxx = (dx**2).sum()*r
        Cxy = (dx*dy).sum()*r
        Cxy = (dy**2).sum()*r
    else:
        # Account for signal-dependent variance.
        Cxx = (q*dx**2).sum()*r
        Cxy = (q*dx*dy).sum()*r
        Cyy = (q*dy**2).sum()*r

    # Return the centroid coordinates and their covariance.
    return (xc, yc, Cxx, Cyy, Cxy)

def invert_2x2_symmetric_matrix(Axx, Axy, Ayy):
    """Invert a 2×2 symmetric matrix.

    Return a 3-tuple ``(Bxx, Bxy, Byy)`` such that ``[[Bxx, Bxy], [Bxy, Byy]]`` is the
    inverse of ``[[Axx, Axy], [Axy, Ayy]]``.

    """
    den = Axx*Ayy - Axy**2
    return Ayy/den, -Axy/den, Axx/den

def _bench0(n=10000000):
    t0 = time.time()
    for i in range(n):
        continue
    return (time.time() - t0)/n

def _bench1(n=10000000):
    t0 = time.time()
    for i in range(n):
        assert i >= 0
    return (time.time() - t0)/n

# With check=False, covariance=False
# ≈ 4.6 μs for a 11×11 ROI in float32 -> ≈ 130 Gflops/s
#
# With check=True, covariance=False
# ≈ 6.5 μs for a 11×11 ROI in float32 without `broadcast_arrays` -> ≈ 93 Gflops/s
# ≈ 8.5 μs with `broadcast_arrays`
#
# With check=False, covariance=True
# ≈ 13.4 μs or a 11×11 ROI in float32
#
# With check=True, covariance=True
# ≈ 15.3 μs for a 11×11 ROI in float32 without `broadcast_arrays` -> ≈ 103 Gflops/s
# ≈ 17.5 μs with `broadcast_arrays`
def _bench2(w, x, y, *, n=100000, check=True, covariance=False):
    t0 = time.time()
    for i in range(n):
        center_of_gravity(w, x, y, check=check, covariance=covariance)
    return (time.time() - t0)/n

def _bench3(w, x, y, *, n=100000):
    t0 = time.time()
    for i in range(n):
        cond = w.min() < 0
    return (time.time() - t0)/n
