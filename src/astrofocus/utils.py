# -*- coding: utf-8 -*-

"""Utility functions."""

import numpy as _np
from numpy import conj, real, imag, log, sqrt, ones, zeros, empty

rand = _np.random.random_sample
randn = _np.random.standard_normal

# Aliases/shortcuts for FFT routines assuming real-valued quantities in the
# direct spaced.
fftshift = _np.fft.fftshift
ifftshift = _np.fft.ifftshift
fft = _np.fft.fftn
def ifft(x): return _np.real(_np.fft.ifftn(x))

def abs2(x):
    """Returns the squared absolute value of its argument."""
    if _np.iscomplexobj(x):
        x_re = x.real
        x_im = x.imag
        return x_re*x_re + x_im*x_im
    else:
        return x*x

def normalize(psf):
    s = sum(psf.ravel())
    if s == 1.0: return psf
    if s > 0.0: return (1.0/s)*psf
    raise ValueError('PSF cannot be normalized: `sum(PSF) = {0}`'.format(s))

def zeropad(arr, shape):
    """
    Zero-pad array `arr` to given `shape`. The contents of `arr` is centered in
    the result according to the conventions used in `fftshift`.
    """
    rank = arr.ndim
    if len(shape) != rank:
        raise ValueError('bad number of dimensions')
    inp_shape = _np.asarray(arr.shape)
    out_shape = _np.asarray(shape)
    if (out_shape < inp_shape).any():
        raise ValueError('output dimensions must be larger or equal input dimensions')
    offset = (out_shape // 2) - (inp_shape // 2)
    z = _np.zeros(shape, dtype=arr.dtype)
    if rank == 0:
        return z
    i0 = offset[0]
    j0 = i0 + inp_shape[0]
    if rank == 1:
        z[i0:j0] = arr
        return z
    i1 = offset[1]
    j1 = i1 + inp_shape[1]
    if rank == 2:
        z[i0:j0,i1:j1] = arr
        return z
    i2 = offset[2]
    j2 = i2 + inp_shape[2]
    if rank == 3:
        z[i0:j0,i1:j1,i2:j2] = arr
        return z
    i3 = offset[3]
    j3 = i3 + inp_shape[3]
    if rank == 4:
        z[i0:j0,i1:j1,i2:j2,i3:j3] = arr
        return z
    i4 = offset[4]
    j4 = i4 + inp_shape[4]
    if rank == 5:
        z[i0:j0,i1:j1,i2:j2,i3:j3,i4:j4] = arr
        return z
    i5 = offset[5]
    j5 = i5 + inp_shape[5]
    if rank == 6:
        z[i0:j0,i1:j1,i2:j2,i3:j3,i4:j4,i5:j5] = arr
        return z
    raise ValueError('too many dimensions')

def DtD(x):
    """
    Return the result of D'⋅D⋅x where D is a (multi-dimensional) finite
    difference operator and D' is its transpose.
    """
    dims = x.shape
    y = np.zeros(dims, dtype=x.dtype) # to store the result
    rank = x.ndim # number of dimensions
    if rank == 0: return y
    if dims[0] >= 2:
        d = x[1:,...] - x[0:-1,...]
        y[1:,...] += d
        y[0:-1,...] -= d
    if rank == 1: return y
    if dims[1] >= 2:
        d = x[:,1:,...] - x[:,0:-1,...]
        y[:,1:,...] += d
        y[:,0:-1,...] -= d
    if rank == 2: return y
    if dims[2] >= 2:
        d = x[:,:,1:,...] - x[:,:,0:-1,...]
        y[:,:,1:,...] += d
        y[:,:,0:-1,...] -= d
    if rank == 3: return y
    if dims[3] >= 2:
        d = x[:,:,:,1:,...] - x[:,:,:,0:-1,...]
        y[:,:,:,1:,...] += d
        y[:,:,:,0:-1,...] -= d
    if rank == 4: return y
    if dims[4] >= 2:
        d = x[:,:,:,:,1:,...] - x[:,:,:,:,0:-1,...]
        y[:,:,:,:,1:,...] += d
        y[:,:,:,:,0:-1,...] -= d
    if rank == 5: return y
    raise ValueError('too many dimensions')
