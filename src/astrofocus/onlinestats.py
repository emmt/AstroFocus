# -*- coding: utf-8 -*-

"""Implement online sample statistics."""

import astropy.io.fits as fits
import copy
import numpy as np

# TODO merge 2 sample statistics
class SampleStatistics:
    """Object to accumulate 2nd order sample statistics.

    ``stat = SampleStatistics()``

    Return an empty object which may updated by providing independent data
    samples as they become available with:

    ``stat.update(data, mask=None)``.

    """
    def __init__(self):
        self._cnt = 0
        self._avg = 0
        self._var = 0

    def __repr__(self):
        cnt = self._cnt
        if np.isscalar(cnt):
            nobs = cnt
        else:
            nobs = np.asarray(cnt).max()
        descr = f'SimpleStatistics with {nobs} observation(s)'
        avg = self._avg
        if not np.isscalar(avg):
            shape = np.asarray(avg).shape
            descr += f' of shape {shape}'
        return descr

    def __copy__(self):
        other = SampleStatistics()
        other._cnt = copy.copy(self._cnt)
        other._avg = copy.copy(self._avg)
        other._var = copy.copy(self._var)
        return other

    def __deepcopy__(self, memo):
        other = SampleStatistics()
        other._cnt = copy.deepcopy(self._cnt, memo)
        other._avg = copy.deepcopy(self._avg, memo)
        other._var = copy.deepcopy(self._var, memo)
        return other

    def __eq__(self, other):
        # TODO Deal with NaNs and missing samples.
        return (other is self) or (isinstance(other, SampleStatistics) and
                                   np.all(other._cnt == self._cnt) and
                                   np.all(other._avg == self._avg) and
                                   np.all(other._var == self._var))

    def count(self):
        """Return the number of observations accumulated in `self`."""
        return self._cnt

    def mean(self):
        """Return the sample mean of the observations accumulated in `self`."""
        return np.asarray(self._avg)

    def var(self, corrected=False):
        """Return the sample variance of the observations accumulated in
        `self`.

        If `corrected` is `True`, an unbiased estimate of the variance
        returned; otherwise, a maximum likelihood estimator is returned.

        """
        var = np.asarray(self._var)
        if corrected:
            n = self._cnt
            return (n / (n - 1)) * var
        else:
            return var

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

        ``self.update(data, where=~np.isnan(data))``

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

    def writeto(self, filename, *, overwrite=False):
        """Write sample statistics to a FITS file."""
        hdr = fits.Header()
        hdr['HDUNAME'] = ('SAMPLE_STATISTICS', 'This HDU contains sample statistics')
        if np.isscalar(self._cnt):
            nobs = self._cnt
        elif np.all(self._cnt == self._cnt.ravel()[0]):
            nobs = self._cnt.ravel()[0]
        else:
            nobs = None
        avg = self.mean()
        data = np.empty((nobs is None and 3 or 2,) + avg.shape, avg.dtype)
        if nobs is None:
            data[0, ...] = self._cnt
            offset = 1
        else:
            hdr['NOBS'] = (nobs, 'Number of observations')
            offset = 0
        data[offset + 0, ...] = avg
        data[offset + 1, ...] = self._var
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        hdu.writeto(filename, overwrite=overwrite)
        return None

    @staticmethod
    def readfrom(filename):
        """Read sample statistics from a FITS file."""
        with fits.open(filename) as hdus:
            for hdu in hdus:
                hdr = hdu.header
                if 'EXTNAME' in hdr:
                    extname = hdr['EXTNAME']
                elif 'HDUNAME' in hdr:
                    extname = hdr['HDUNAME']
                else:
                    extname = None
                if extname != 'SAMPLE_STATISTICS':
                    continue
                data = hdu.data
                shape = data.shape
                stat = SampleStatistics()
                if 'NOBS' not in hdr:
                    # Non-uniform number of observations.
                    if shape[0] != 3:
                        raise Exception('Expecting NAXIS1=3, got NAXIS1={0}'.format(shape[-1]))
                    cnt = data[0,...].astype('int')
                    cnt0 = cnt.ravel()[0]
                    stat._cnt = (cnt == cnt0).all() and cnt0 or cnt
                    offset = 1
                else:
                    # Uniform number of observations.
                    if shape[0] != 2:
                        raise Exception('Expecting NAXIS1=2, got NAXIS1={0}'.format(shape[-1]))
                    stat._cnt = int(hdr['NOBS'])
                    offset = 0
                stat._avg = data[offset + 0,...]
                stat._var = data[offset + 1,...]
                return stat

        raise Exception('"SAMPLE_STATISTICS" extension not found in FITS file')
