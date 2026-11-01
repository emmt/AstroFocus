# -*- coding: utf-8 -*-

"""Utilities for dealing with images (any 2-dimensional array)."""

import copy
import numpy as np

class ImageROI(object):
    """Image region of interest.

    The region of interest (ROI) is a rectangular box whose sides are aligned
    with the axes of the image (any 2-dimensional array). The index ranges of
    the region of interest can be specified by different combination of
    keywords among:

    - `xmin`, `xmax`, `ymin, and `ymax`, the inclusive bounds of the region.

    - `xcen`, `ycen`, `width`, and `height`, the central coordinates and size
      of the region.

    Axes `x` and `y` respectively correspond the 2nd and 1st indices (as for
    images displayed by `imshow` with default settings). The range of interest
    along axis `x` is specified by 2 keywords out of `xmin`, `xmax`, `xcen`, or
    `width`. The range of interest along axis `y` is specified by 2 keywords
    out of `ymin`, `ymax`, `ycen`, or `height`.

    If a range is specified by its center `c` and its size `s` (`xcen` and
    `width` or `ycen` and `height`), the range of interest will have a length
    `n = round(s)` and will start at `i = round(c - (n - 1)/2)` and stop at `j
    = i + n - 1` (inclusive).

    In other cases, parameter `xcen` and `ycen`, if specified, are rounded to
    the nearest half-integer, while other parameters are rounded to the nearest
    integer.

    An image ROI `roi` is a callable object to crop an image `img`:
    ``roi(img)`` is the same as ``roi.get(img)``.

    """
    def __init__(self, *,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 xcen=None, ycen=None, width=None, height=None):
        # Compute ROI limits (min. is inclusive, end is exclusive).
        beg2, end2 = _parse_range('x', xmin, xmax, xcen, width)
        self._beg2 = beg2
        self._end2 = end2
        beg1, end1 = _parse_range('y', ymin, ymax, ycen, height)
        self._beg1 = beg1
        self._end1 = end1

    def __repr__(self):
        return 'ImageROI(xmin={}, xmax={}, ymin={}, ymax={})'.format(
            self._beg2, self._end2 - 1, self._beg1, self._end1 - 1)

    def __eq__(self, other):
        if other is self:
            return True
        elif not isinstance(other, ImageROI):
            return False
        elif self.is_empty():
            return other.is_empty()
        else:
            return (other._beg1 == self._beg1 and other._end1 == self._end1 and
                    other._beg2 == self._beg2 and other._end2 == self._end2)

    def __call__(self, img):
        return self.get(img)

    def copy(self):
        """Return an independent copy of the image ROI."""
        return copy.copy(self)

    def crop(self, img):
        """Extract the pixels of the ROI of image `img` as an independent array."""
        return copy.copy(np.asarray(img)[self._beg1:self._end1, self._beg2:self._end2])

    def get(self, img):
        """Return a view on the pixels of the ROI of image `img`."""
        return np.asarray(img)[self._beg1:self._end1, self._beg2:self._end2]

    def set(self, img, rhs):
        """Set the pixels of the ROI of image `img` to `rhs`."""
        assert isinstance(img, np.ndarray)
        img[self._beg1:self._end1, self._beg2:self._end2] = rhs
        return None

    def add(self, img, rhs):
        """Add in-place `rhs` to the pixels in the ROI of image `img`."""
        assert isinstance(img, np.ndarray)
        img[self._beg1:self._end1, self._beg2:self._end2] += rhs
        return None

    def subtract(self, img, rhs):
        """Subtract in-place `rhs` to the pixels in the ROI of image `img`."""
        assert isinstance(img, np.ndarray)
        img[self._beg1:self._end1, self._beg2:self._end2] -= rhs
        return None

    def multiply(self, img, rhs):
        """Multiply in-place the pixels in the ROI of image `img` by `rhs`."""
        assert isinstance(img, np.ndarray)
        img[self._beg1:self._end1, self._beg2:self._end2] *= rhs
        return None

    def divide(self, img, rhs):
        """Divide in-place the pixels in the ROI of image `img` by `rhs`."""
        assert isinstance(img, np.ndarray)
        img[self._beg1:self._end1, self._beg2:self._end2] /= rhs
        return None

    def xmin(self):
        """Return the first index along axis `x` (2nd axis) of the image ROI."""
        return self._beg2

    def xmax(self):
        """Return the last index along axis `x` (2nd axis) of the image ROI."""
        return self._end2 - 1

    def xcen(self):
        """Return the central position along axis `x` (2nd axis) of the image ROI."""
        return (self._beg2 + self._end2 - 1)/2

    def xrange(self):
        """Return the range of indices along axis `x` (2nd axis) of the image ROI."""
        return range(self._beg2, self._end2)

    def width(self):
        """Return the number of indices along axis `x` (2nd axis) of the image ROI."""
        return max(self._end2 - self._beg2, 0)

    def ymin(self):
        """Return the first index along axis `y` (1st axis) of the image ROI."""
        return self._beg1

    def ymax(self):
        """Return the last index along axis `y` (1st axis) of the image ROI."""
        return self._end1 - 1

    def ycen(self):
        """Return the central position along axis `y` (1st axis) of the image ROI."""
        return (self._beg1 + self._end1 - 1)/2

    def yrange(self):
        """Return the range of indices along axis `y` (1st axis) of the image ROI."""
        return range(self._beg1, self._end1)

    def height(self):
        """Return the number of indices along axis `y` (1st axis) of the image ROI."""
        return max(self._end1 - self._beg1, 0)

    def is_empty(self):
        """Return whether the image ROI is empty."""
        return not (self._beg1 < self._end1 and self._beg2 < self._end2)

    def is_part_of(self, other):
        """Return whether the image ROI is entirely inside another ROI or image."""
        if isinstance(other, ImageROI):
            if other is self or self.is_empty():
                return True
            else:
                return (other._beg1 <= self._beg1 and self._end1 <= other._end1 and
                        other._beg2 <= self._beg2 and self._end2 <= other._end2)
        elif isinstance(other, np.ndarray) and other.ndim == 2:
            if self.is_empty():
                return True
            else:
                other_end1, other_end2 = other.shape
                return (0 <= self._beg1 and self._end1 <= other_end1 and
                        0 <= self._beg2 and self._end2 <= other_end2)
        return False

    def intersect_with(self, other):
        """Return an image ROI which is the intersection of the image ROI with
        another image ROI or with the indices of an image (any 2-dimensional
        array)."""
        if isinstance(other, ImageROI):
            other_beg1 = other._beg1
            other_end1 = other._end1
            other_beg2 = other._beg2
            other_end2 = other._end2
        elif isinstance(other, np.ndarray) and other.ndim == 2:
            other_end1, other_end2 = other.shape
            other_beg1, other_beg2 = 0, 0
        else:
            raise ValueError("expecting an image ROI or a 2-dimensional array")
        r = copy.copy(self)
        r.beg1, r.end1 = _intersect(self._beg1, self._end1, other_beg1, other_end1)
        r.beg2, r.end2 = _intersect(self._beg2, self._end2, other_beg2, other_end2)
        return r

    def move_by(self, dx, dy):
        """Move image ROI by `(round(dx), round(dy))` and return it.

        Operation is done in-place, call ``self.copy().move_by(dx, dy)`` to not
        modify `self`.

        """
        if not self.is_empty():
            dx = round(dx)
            self._beg2 += dx
            self._end2 += dx
            dy = round(dy)
            self._beg1 += dy
            self._end1 += dy
        return self

    def center_at(self, x, y):
        """Center image ROI at `(x, y)` and return it.

        Operation is done in-place, call ``self.copy().center_at(x, y)`` to not
        modify `self`.

        """
        return self.move_by(x - self.xcen(), y - self.ycen())

    def grow_by(self, dx, dy=None):
        """Grow image ROI by `(round(dx), round(dy))` pixels on all sides and
        return it.

        If `dy` is not specified (or `None`), `dx = dy` is assumed. For a
        negative adjustment, the ROI is shrunk along the correspond axis.

        Operation is done in-place, call ``self.copy().grow_by(dx, dy)`` to not
        modify `self`.

        """
        if not self.is_empty():
            dx = round(dx)
            self._beg2 -= dx
            self._end2 += dx
            if dy is None:
                dy = dx
            else:
                dy = round(dy)
            self._beg1 -= dy
            self._end1 += dy
        return self

    def shrink_by(self, dx, dy=None):
        """Shrink image ROI by `(round(dx), round(dy))` pixels on all sides and
        return it.

        If `dy` is not specified (or `None`), `dx = dy` is assumed. For a
        negative adjustment, the ROI is grown along the corresponding axis.

        Operation is done in-place, call ``self.copy().shrink_by(dx, dy)`` to
        not modify `self`.

        """
        if not self.is_empty():
            dx = round(dx)
            self._beg2 += dx
            self._end2 -= dx
            if dy is None:
                dy = dx
            else:
                dy = round(dy)
            self._beg1 += dy
            self._end1 -= dy
        return self

def _parse_range(name, rmin, rmax, rcen, rlen):
    rend = None
    if rmin is not None:
        rmin = round(rmin)
        if rmax is not None and rcen is None and rlen is None:
            rend = round(rmax) + 1
        elif rmax is None and rcen is not None and rlen is None:
            rend = round(2*rcen) - rmin + 1
        elif rmax is None and rcen is None and rlen is not None:
            rend = rmin + round(rlen)
    elif rmax is not None:
        rmax = round(rmax)
        rend = rmax + 1
        if rcen is not None and rlen is None:
            rmin = round(2*rcen) - rmax
        elif rcen is None and rlen is not None:
            rmin = rend - round(rlen)
    elif rcen is not None and rlen is not None:
        rlen = round(rlen)
        rmin = round(rcen - (rlen - 1)/2)
        rend = rmin + rlen
    if rend is None:
        raise Exception("bad set of keywords for the `{}` range".format(name))
    if rmin < rend:
        return rmin, rend
    else:
        return 0, 0

def _intersect(a_beg, a_end, b_beg, b_end):
    c_beg = max(a_beg, b_beg)
    c_end = min(a_end, b_end)
    if c_beg < c_end: return (c_beg, c_end)
    else:             return (0, 0)
