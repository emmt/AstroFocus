# -*- coding: utf-8 -*-

"""Filters."""

import numpy as np
import math
from .utils import outer_sum, outer_prod

def ball_2d_window(radius):
    """Return a 2-dimensional window of Boolean values corresponding to a ball
    of given radius (rounded to nearest integer)."""
    r = round(radius) # only works correctly for integer radius
    x2 = np.array(range(-r, r + 1))**2
    return outer_sum(x2, x2) <= (r + 0.5)**2

def _normalize_filter(f, normalize=None):
    if normalize is None:
        return f
    nrm = normalize(f)
    if nrm == 1:
        return f
    return f/nrm

GAUSSIAN_FWHM = math.sqrt(math.log(4))
def gaussian_1d_filter(fwhm, tol=1e-3, normalize=None):
    """Return a Gaussian shaped function with given full-width at half-maximum.

    The returned filter kernel is centered and truncated for values smaller
    than `tol` times the peak-to-valley value of the filter.

    Keyword `normalize` can be specified with a function that is applied to the
    filter to normalize its shape. For example, with `normalize = np.max`, the
    peak value of the filter is 1 (this is the default); with `normalize =
    np.sum`, the sum of the filter is 1, etc.

    """
    # The Gaussian shape is `g(x) = exp((-1/2)*(s*x)**2)` with:
    s = GAUSSIAN_FWHM/fwhm # scaling factor
    # Compute half-width `w` of filter array such that `g(w) ≤ tol`
    # ==> w ≥ s*sqrt(-2*log(tol))
    w = math.ceil(s*math.sqrt(-2*math.log(tol)))
    # TODO optimize (compute only half)
    x = np.array(range(-w, w+1))
    f = np.exp((-1/(2*s**2))*x**2)
    return _normalize_filter(f, normalize)

def gaussian_2d_filter(fwhm1, fwhm2=None, normalize=None):
    """Return a 2-dimensional Gaussian shaped filter with given full-width at half-maximum.

    The 2-dimensional filter is separable. If the FWHM along the 2nd dimension
    is not specified, `fwhm2 = fwhm1` is assumed.

    The filter is centered and truncated outside the support of the cubic
    B-spline function.

    """
    f1 = gaussian_1d_filter(fwhm1)
    if fwhm2 is None:
        f2 = f1
    else:
        f2 = gaussian_1d_filter(fwhm2)
    f = outer_prod(f1, f2)
    return _normalize_filter(f, normalize)

CUBIC_BSPLINE_FWHM = 1.444703448928752590331004298512891328576
def cubic_bspline(x):
    """Return the cubic B-spline function for coordinate(s) `x`.

    The cubic B-spline function is defined by:

    |           2/3 + (x/2 - 1)*x^2    if abs(x) ≤ 1
    |    f(x) = 1/6 * (2 - x)^3        if 1 ≤ abs(X) ≤ 2
    |           0                      if abs(x) ≥ 2

    The cubic B-spline has the following properties:

    (1) `f(x)` is exactly zero outside (including boundaries) the support [-2, +2]. `f(x)`
        and its first and second derivatives are continuous functions everywhere and are
        piecewise polynomials.

    (2) `f(x)` is an even unimodal bell-shaped function: `f(-x) = f(x)` and
        `max_x f(x) = f(0) = 2/3`.

    (3) The full-width at half-maximum of the cubic B-spline is:
        `CUBIC_BSPLINE_FWHM = 4/3 * (1 + 2*cos((pi + atan(3*sqrt(7)))/3)) ≈ 1.445`

    (4) The cubic B-spline is normalized:

        |     ╭ +Inf       ╭ +2
        |     |            |
        |     | f(x) dx =  | f(x) dx = 1
        |     |            |
        |     ╯ -Inf       ╯ -2

        and `f(-1) + f(0) + f(+1) = 1`.

    """
    abs_x = np.abs(x)
    y = np.zeros(abs_x.shape, float)
    cond = abs_x < 1
    if cond.any():
        u = abs_x[cond]
        y[cond] = 2/3 + (u/2 - 1)*u**2
    cond = (~cond) & (abs_x < 2)
    if cond.any():
        u = abs_x[cond]
        y[cond] = ((2 - u)**3)/6
    return y

def cubic_bspline_1d_filter(fwhm, normalize=None):
    """Return a cubic B-spline filter with given full-width at half-maximum.

    The filter is centered and truncated outside the support of the cubic
    B-spline function.

    """
    n = math.ceil(fwhm) # half-size
    f = cubic_bspline((CUBIC_BSPLINE_FWHM/fwhm)*np.array(range(-n, n+1)))
    return _normalize_filter(f, normalize)

def cubic_bspline_2d_filter(fwhm1, fwhm2=None, normalize=None):
    """Return a 2-dimensional cubic B-spline filter with given full-width at half-maximum.

    The 2-dimensional filter is separable. If the FWHM along the 2nd dimension
    is not specified, `fwhm2 = fwhm1` is assumed.

    The filter is centered and truncated outside the support of the cubic
    B-spline function.

    """
    f1 = cubic_bspline_1d_filter(fwhm1)
    if fwhm2 is None:
        f2 = f1
    else:
        f2 = cubic_bspline_1d_filter(fwhm2)
    f = outer_prod(f1, f2)
    return _normalize_filter(f, normalize)
