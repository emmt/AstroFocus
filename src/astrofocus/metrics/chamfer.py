# -*- coding: utf-8 -*-

"""Implement chamfer distance for various cases."""

import numpy as np

def offset_to_nearest_grid_node(off, stp):
    """Return the uni-dimensional offset to the nearest grid node.

    Arguments are the offset `off` to any of the grid nodes and the grid step size `stp`
    (both given in the same units).

    The result is in the range ``[-stp/2, +stp/2]``.

    """
    return off - np.round(off/stp)*stp

def index_of_nearest_grid_node(off, stp, ref=None):
    """Return the index of the nearest grid node.

    Arguments are: the offset `off` to some reference grid node, the grid step size `stp`
    (both given in the same units), and the index `ref` of the reference node. If `ref` is
    `None` (the default), the result is relative to the index of the reference node.

    """
    ind = np.round(off/stp).astype(int)
    if ref is None:
        return ind
    else:
        return ind + ref

class ChamferDistanceToGrid(object):
    """Callable object to measure the chamfer distance to a 2-dimensional
    evenly spaced Cartesian grid aligned with the axes of the coordinate frame.

    Constructor arguments are `w` (optional non-negative weights), `x`
    (abscissae of points), and `y` (ordinates of points).

    Allowed keywords are `xoff`, the abscissa of one of the grid nodes, `yoff`,
    the ordinate of one of the grid nodes, `step`, the separation of the grid
    nodes, and `auto_update` a Boolean to specify whether the grid parameters
    should be automatically updated when the object is called as a function
    with grid parameter(s) which improve the chamfer distance.

    Calling the object with an argument `arg` return the squared (weighted)
    chamfer distance for the grid parameterized by `arg`. There are several
    possibilities:

    - If `arg` is `None` (the default). The grid parameters are `self.xoff,
      `self.yoff`, and `self.step`.

    - If `arg` is a scalar or a 1-vector like, it specifies the grid step. The
      grid offset is ``(self.xoff, self.yoff)``.

    - If `arg` is 2-vector like, it specifies the grid offsets: ``xoff =
      arg[0]`` and ``yoff = arg[1]`. The grid step is ``self.step``.

    - If `arg` is 3-vector like, it specifies the grid offsets and step: ``xoff
      = arg[0]``, ``yoff = arg[1]`, and ``step = arg[2]`.

    Attribute `ncalls` gives the number a calls to the object as a function.

    """
    def __init__(self, *args, **kwds):
        if len(args) == 2:
            w = None
            x, y = np.broadcast_arrays(*args)
        elif len(args) == 3:
            w, x, y = np.broadcast_arrays(*args)
        else:
            raise Exception('expecting 2 or 3 arguments: `w`, `x`, and `y` with `w` optional')
        xoff = 0.0
        yoff = 0.0
        step = 1.0
        auto_update = False
        for key in kwds:
            val = kwds[key]
            if key == 'xoff':
                xoff = val
            elif key == 'yoff':
                yoff = val
            elif key == 'step':
                step = val
            elif key == 'auto_update':
                if not isinstance(val, bool):
                    raise TypeError('value of `auto_update` must be a Boolean')
                auto_update = val
            else:
                raise KeyError('invalid keyword `{}`'.format(key))
        self.ncalls = 0
        self.lsd = np.inf
        self.auto_update = auto_update
        self.xoff = xoff
        self.yoff = xoff
        self.step = step
        self.w = w
        self.x = x
        self.y = y

    def reset(self):
        """Reset the internal counter of number of calls to the object as a
        function and return the former number of calls."""
        ncalls = self.ncalls
        self.ncalls = 0
        return ncalls

    def get_offsets(self):
        """Get the grid offset, that is the coordinates of one of the grid
        nodes.

        """
        return (self.xoff, self.yoff)

    def set_offsets(self, xoff, yoff):
        """Set the grid offset, that is the coordinates of one of the grid
        nodes.

        """
        self.xoff = xoff
        self.yoff = yoff

    def get_step(self):
        """Get the grid step."""
        return self.step

    def set_step(self, step):
        """Set the grid step."""
        self.step = step

    def get_mean_distance(self, *args):
        """Return the mean Euclidean distance of the points to the grid."""
        num = self(*args)
        if self.w is None:
            den = len(self.x)
        else:
            den = self.w.sum()
        return np.sqrt(num/den)

    def __call__(self, arg=None):
        # if arg is None and self.auto_update and np.isfinite(self.lsd):
        #     # Save computations when the squared distance is cached.
        #     # FIXME This is potentially wrong if auto-update is turned on
        #     #       and computations not yet done.
        #     return self.lsd
        if np.isscalar(arg):
            xoff = self.xoff
            yoff = self.yoff
            step = arg
        elif arg is None:
            xoff = self.xoff
            yoff = self.yoff
            step = self.step
        else:
            n = len(arg)
            if n == 1:
                xoff = self.xoff
                yoff = self.yoff
                step = arg[0]
            elif n == 2:
                xoff = arg[0]
                yoff = arg[1]
                step = self.step
            elif n == 3:
                xoff = arg[0]
                yoff = arg[1]
                step = arg[2]
            else:
                raise ValueError('argument must be a scalar or a n-vector like with n = 1, 2, or 3')
        q = (offset_to_nearest_grid_node(self.x - xoff, step)**2 +
             offset_to_nearest_grid_node(self.y - yoff, step)**2)
        w = self.w
        sd = (w is None and q or w*q).sum()
        self.ncalls += 1
        if self.auto_update and sd < self.lsd:
            # Memorize the least squared distance (LSD) and its parameters.
            self.lsd = sd
            self.xoff = xoff
            self.yoff = yoff
            self.step = step
        return sd
