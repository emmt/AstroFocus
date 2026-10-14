# -*- coding: utf-8 -*-

"""Useful functions for numerical optimization."""

# Insure compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as _np # to deal with numerical arrays

def inner(x,y):
    """Compute inner product of X and Y regardless their shapes
        (their number of elements must however match)."""
    return _np.inner(x.ravel(),y.ravel())
