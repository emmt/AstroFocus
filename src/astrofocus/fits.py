# -*- coding: utf-8 -*-

"""Reading and writing of FITS files."""

import astropy.io.fits as fits

def readfits(filename):
    """Read a FITS image from a file."""
    with fits.open(filename) as hdus:
        return hdus[0].data

def writefits(filename, data, comment=None, history=None, overwrite=False):
    """Write an image into a FITS file."""
    hdr = fits.Header()
    if comment is not None:
        hdr['COMMENT'] = comment
    if history is not None:
        hdr['HISTORY'] = history
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(filename, overwrite=overwrite)

def add_comment(hdr, *args):
    add_commentary(hdr, 'COMMENT', *args)

def add_history(hdr, *args):
    add_commentary(hdr, 'HISTORY', *args)

def add_commentary(hdr, key, *args):
    for arg in args:
        if isinstance(arg, (tuple, list)):
            for val in arg:
                add_commentary1(hdr, key, val)
        else:
            add_commentary1(hdr, key, arg)

def add_commentary1(hdr, key, val):
    """Add a single commentary FITS card."""
    if isinstance(val, str):
        hdr[key] = val
    elif val is not None:
        raise ValueError(f'unexpected value of type {type(val)} for FITS {key}')
