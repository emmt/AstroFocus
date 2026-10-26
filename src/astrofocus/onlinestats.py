# -*- coding: utf-8 -*-

"""Implement online sample statistics."""

import astropy.io.fits as fits
import numpy as np

# TODO merge 2 sample statistics
class SampleStatistics:
    """Object to accumulate 2nd order sample statistics.

    ``stat = SampleStatistics()``

    Return an empty object which may updated by providing independent data
    samples as they become available with: ``stat.update(data, mask=None)``.

    ``stat = SimpleStatistics(filename)``

    Return an object with content read from file `filename`.

    """
    def __init__(self, filename=None):
        self._cnt = 0
        self._avg = 0
        self._var = 0
        if filename is not None:
            self.readfrom(filename)

    def _clone(self, op=copy):
        other = SampleStatistics()
        other._cnt = op(self._cnt)
        other._avg = op(self._avg)
        other._var = op(self._var)
        return other

    def __copy__(self):
        return _clone(self, copy)

    def __deepcopy__(self):
        return _clone(self, deepcopy)

    def count(self):
        """Return the number of observations accumulated in `self`."""
        return self._cnt

    def mean(self):
        """Return the sample mean of the observations accumulated in `self`."""
        return self._avg

    def var(self, corrected=False):
        """Return the sample variance of the observations accumulated in
        `self`.

        If `corrected` is `True`, an unbiased estimate of the variance
        returned; otherwise, a maximum likelihood estimator is returned.

        """
        if corrected:
            n = self._cnt
            return (n / (n - 1)) * self._var
        else:
            return self._var

    def std(self, corrected=False):
        """Return the sample standard deviation of the observations accumulated
        in `self`.

        This is equivalent to ``np.sqrt(self.var(corrected))``.

        """
        return np.sqrt(self.var(corrected))

    def wgt(self, corrected=False):
        """Return the element-wise precision (a.k.a. statistical weights) of
        the observations accumulated in `self`.

        This is equivalent to ``1 / self.var(corrected)``.

        """
        return 1 / self.var(corrected)

    def update(self, data, where=None):
        """Add a new observed sample `data` to the observations accumulated in
        `self`.

        Optional argument `where` indicates the valid entries of `data` (those
        for which `where` is non-zero). If `where` is `None` (the default), it
        is assumed that all entries in `data` are valid. If `where` is neither
        `None`, `True, nor `False`, it is broadcast to the shape of `data`. For
        example, to exclude NaNs:

        ``self.update(date, where=~np.isnan(data))``

        """
        if where is None or np.all(where):
            # It is assumed that all samples count for 1.
            n = self._cnt
            m = self._avg
            v = self._var
            np1 = n + 1
            u = (data - m)/np1
            self._cnt = np1
            self._avg = m + u
            self._var = n*(v/np1 + np.square(u))
        elif np.any(where):
            # Some (but not all) data entries are valid.
            shape = data.shape
            where = np.asarray(where)
            if where.dtype != bool:
                where = where.astype(bool)
            if where.shape != shape:
                where = np.broadcast_to(where, shape)
            if np.isscalar(self._cnt):
                self._cnt = np.broadcast_to(self._cnt, shape)
            if np.isscalar(self._avg):
                self._avg = np.broadcast_to(self._avg, shape)
            if np.isscalar(self._var):
                self._var = np.broadcast_to(self._var, shape)
            n = self._cnt[where]
            m = self._avg[where]
            v = self._var[where]
            np1 = n + 1
            u = (data[where] - m)/np1
            self._cnt[where] = np1
            self._avg[where] = m + u
            self._var[where] = n*(v/np1 + np.square(u))
        return None

    def writeto(self, filename):
        """Write sample statistics to a FITS file."""
        hdr = fits.Header()
        hdr['HDUNAME'] = ('SAMPLE_STATISTICS', 'This HDU contains sample statistics')
        if np.isscalar(self._cnt):
            nobs = self._cnt
        elif np.all(self._cnt == self._cnt.ravel()[0]):
            nobs = self._cnt.ravel()[0]
        else:
            nobs = None
        data = np.empty((nobs is None and 3 or 2,) + self._avg.shape, self._avg.dtype)
        if nobs is None:
            data[0, ...] = self._cnt
            offset = 1
        else:
            hdr['NOBS'] = (nobs, 'Number of observations')
            offset = 0
        data[offset + 0, ...] = self._avg
        data[offset + 1, ...] = self._var
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdu.writeto(filename)
        return None

    def readfrom(self, filename):
        """Read sample statistics from a FITS file.

        Prior content of `self`, if any, is replaced by the statistics read
        from `filename`.

        """
        found = False
        with fits.open(filename) as hdus:
            for hdu in hdus:
                hdr = hdu.header
                if 'HDUNAME' not in hdr or hdr['HDUNAME'] != 'SAMPLE_STATISTICS':
                    continue
                data = hdu.data
                shape = data.shape
                if 'NOBS' not in hdr:
                    # Non-uniform number of observations.
                    if shape[0] != 3:
                        raise Exception('Expecting NAXIS1=3, got NAXIS1={0}'.format(shape[-1]))
                    cnt = data[0,...].astype('int')
                    cnt0 = cnt.ravel()[0]
                    self._cnt = (cnt == cnt0).all() and cnt0 or cnt
                    offset = 1
                else:
                    # Uniform number of observations.
                    if shape[0] != 2:
                        raise Exception('Expecting NAXIS1=2, got NAXIS1={0}'.format(shape[-1]))
                    self._cnt = int(hdr['NOBS'])
                    offset = 0
                self._avg = data[offset + 0,...]
                self._var = data[offset + 1,...]
                found = True
                break
        if not found:
            raise Exception('HDUNAME="SAMPLE_STATISTICS" not found in FITS file')
        return None
