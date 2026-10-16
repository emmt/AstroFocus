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
    raise ValueError("PSF cannot be normalized: `sum(PSF) = {0}`".format(s))

def zeropad(arr, shape):
    """
    Zero-pad array `arr` to given shape. The contents of `arr` is approximately
    centered in the result.
    """
    rank = arr.ndim
    if len(shape) != rank:
        raise ValueError("bad number of dimensions")
    inp_shape = _np.asarray(arr.shape)
    out_shape = _np.asarray(shape)
    if (out_shape < inp_shape).any():
        raise ValueError("output dimensions must be larger or equal input dimensions")
    offset = (out_shape // 2) - (inp_shape // 2)
    z = _np.zeros(shape, dtype=arr.dtype)
    if rank == 1:
        i0 = offset[0]; n0 = i0 + inp_shape[0]
        z[i0:n0] = arr
    elif rank == 2:
        i0 = offset[0]; n0 = i0 + inp_shape[0]
        i1 = offset[1]; n1 = i1 + inp_shape[1]
        z[i0:n0,i1:n1] = arr
    elif rank == 3:
        i0 = offset[0]; n0 = i0 + inp_shape[0]
        i1 = offset[1]; n1 = i1 + inp_shape[1]
        i2 = offset[2]; n2 = i2 + inp_shape[2]
        z[i0:n0,i1:n1,i2:n2] = arr
    elif rank == 4:
        i0 = offset[0]; n0 = i0 + inp_shape[0]
        i1 = offset[1]; n1 = i1 + inp_shape[1]
        i2 = offset[2]; n2 = i2 + inp_shape[2]
        i3 = offset[3]; n3 = i3 + inp_shape[3]
        z[i0:n0,i1:n1,i2:n2,i3:n3] = arr
    elif rank == 5:
        i0 = offset[0]; n0 = i0 + inp_shape[0]
        i1 = offset[1]; n1 = i1 + inp_shape[1]
        i2 = offset[2]; n2 = i2 + inp_shape[2]
        i3 = offset[3]; n3 = i3 + inp_shape[3]
        i4 = offset[4]; n4 = i4 + inp_shape[4]
        z[i0:n0,i1:n1,i2:n2,i3:n3,i4:n4] = arr
    elif rank == 6:
        i0 = offset[0]; n0 = i0 + inp_shape[0]
        i1 = offset[1]; n1 = i1 + inp_shape[1]
        i2 = offset[2]; n2 = i2 + inp_shape[2]
        i3 = offset[3]; n3 = i3 + inp_shape[3]
        i4 = offset[4]; n4 = i4 + inp_shape[4]
        i5 = offset[5]; n5 = i5 + inp_shape[5]
        z[i0:n0,i1:n1,i2:n2,i3:n3,i4:n4,i5:n5] = arr
    else:
        raise ValueError("too many dimensions")
    return z

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
