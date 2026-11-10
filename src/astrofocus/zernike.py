# -*- coding: utf-8 -*-

"""Zernike polynomials."""

import math
import numpy as np

def zernike_count(n):
    """Return the number of Zernike polynomials of radial index ``≤ n``."""
    if not isinstance(n, int):
        n = math.floor(n)
    if n < 0:
        return 0
    return (n + 1)*(n + 2)//2

def zernike_index(m, n):
    """Return the 0-based index of the Zernike polynomial ``Z[m,n]``
       in the list of Zernike polynomials."""
    if not isinstance(m, int):
        raise TypeError('`m` must be integer')
    if not isinstance(n, int):
        raise TypeError('`n` must be integer')
    npm = n + m
    if abs(m) > n or (npm & 1) != 0:
        raise TypeError('`n - |m|` must be non-negative and even')
    return n*(n + 1)//2 + (n + m)//2

def zernike_radial_index(m, n):
    """Return the 0-based index of Zernike radial polynomial ``R[m,n]``."""
    if not isinstance(m, int):
        raise TypeError('`m` must be integer')
    if not isinstance(n, int):
        raise TypeError('`n` must be integer')
    if abs(m) > n or ((n + m) & 1) != 0:
        raise TypeError('`n - |m|` must be non-negative and even')
    return _radial_index(m, n)

def _radial_index(m, n):
    """Return the 0-based index of Zernike radial polynomial ``R[m,n]``.

    This private function assumes arguments are valid. """

    # Azimuthal modes are stored in decreasing azimuthal order for each radial
    # order and there are ``1 + n + (n*n)//4`` number of radial modes of degree
    # ≤ n in total.
    return n + (n*n)//4 - abs(m)//2


def zernike_polynonials(x, y, nmax):
    """Return a list of all Zernike polynonial of degree ``n ≤ nmax`` for
    Cartesian coordinates ``(x,y)``."""

    # Check arguments.
    if not isinstance(nmax, int) or nmax < 0:
        raise ValueError('`nmax` must be a non-negative integer')

    # Infer shape and floating type for computations. Convert x and y if needed.
    x = np.asarray(x)
    y = np.asarray(y)
    shape = np.broadcast(x, y).shape
    Tx = x.dtype.type
    Ty = y.dtype.type
    T = type(Tx(True)/Ty(True))
    if Tx is not T:
        x = x.astype(T)
    if Ty is not T:
        y = y.astype(T)

    # Zernike polynomial for n=0 (piston).
    ones = np.ones(shape, T)
    Z = [ones]
    if nmax == 0:
        return Z

    # Zernike polynomials for n=1 (tip-tilt).
    x, y = np.broadcast_arrays(x, y)
    Z.append(x)
    Z.append(y)
    if nmax == 1:
        return Z

    # For higher orders, polar coordinates are needed.
    r = np.hypot(x, y)
    cond = np.logical_not(r)
    if cond.any:
        # Avoid division by zero.
        den = r + cond
        cos_phi = x/den
        sin_phi = y/den
        cos_phi[cond] = 1
    else:
        cos_phi = x/r
        sin_phi = y/r

    # Tables for `r**m`, `sin(m*phi)`, and `cos(m*phi)`.
    zeros = np.zeros(shape, T)
    r_pow = [ones, r]
    sin_phi_times = [zeros, sin_phi]
    cos_phi_times = [ones, cos_phi]

    # Table of radial modes.
    R = [ones, r]

    # Initialize recurrence.
    r_pow_2 = r**2
    sin_nphi = sin_phi
    cos_nphi = cos_phi
    for n in range(2, nmax + 1):
        # Update radius and trigonometric tables.
        if n == 2:
            r_pow_n = r_pow_2
        else:
            # NOTE not `*=` because we want that the result be a different array
            r_pow_n = r_pow_n*r
        r_pow.append(r_pow_n)
        cos_nphi, sin_nphi = (cos_phi*cos_nphi - sin_phi*sin_nphi,
                              cos_phi*sin_nphi + sin_phi*cos_nphi)
        sin_phi_times.append(sin_nphi)
        cos_phi_times.append(cos_nphi)

        # Store modes for `m = n, n-2, ..., -n`.
        for m in range(n, - n - 1, -2):
            if m < 0:
                R_mn = R[_radial_index(m, n)]
                Z_mn = R_mn*sin_phi_times[abs(m)]
            else:
                # Compute radial Zernike polynomial.
                if n == m:
                    R_mn = r_pow[m]
                elif n == m + 2:
                    R_mn = ((m + 2)*r_pow_2 - (m + 1))*r_pow[m]
                else:
                    den = (n + m)*(n - m)*(n - 2)
                    a = T(4*n*(n - 1)*(n - 2)/den)
                    b = T(2*(n - 1)*(m*m + n*(n - 2))/den)
                    c = T(n*(n - 2 + m)*(n - 2 - m)/den)
                    R_mn = ((a*r_pow_2 - b)*R[_radial_index(m, n - 2)]
                            - c*R[_radial_index(m, n - 4)])
                R.append(R_mn) # update table of radial Zernike polynomials
                Z_mn = R_mn*cos_phi_times[m]
            Z.append(Z_mn)
    return Z
