# -*- coding: utf-8 -*-

"""Reading and writing of FITS files."""

import astropy.io.fits as _fits

def readfits(filename):
    """Read a FITS image from a file."""
    with _fits.open(filename) as hdus:
        return hdus[0].data

def writefits(filename, data, comment=None, history=None):
    """Write an image into a FITS file."""
    hdr = _fits.Header()
    if comment is not None:
        hdr['COMMENT'] = comment
    if history is not None:
        hdr['HISTORY'] = history
    hdu = _fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(filename)
