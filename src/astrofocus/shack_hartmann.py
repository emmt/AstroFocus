# -*- coding: utf-8 -*-

"""Implement methods for Shack-Hartmann wavefront sensors."""

import math
import scipy

import matplotlib.pyplot as plt
import numpy as np

from .centroids import center_of_gravity
from .metrics import ChamferDistanceToGrid
from .filters import gaussian_2d_filter, cubic_bspline_2d_filter, ball_2d_window
from .images import ImageROI
from .utils import Result, vector_like_length

class ShackHartmannGrid:
    """Shack-Hartmann grid.

    Mandatory attributes are:

    - `.step` the spacing oth grid nodes (in pixels).

    - `.origin` the coordinates of the first grid node (in pixels).

    - `.x`, `.y`, and `.z`, each a 2-dimensional array of the shape of the
      grid: ``.x[i,j]``, ``.y[i,j]``, and ``.z[i,j]`` are the coordinates and
      value of the peak of the spot found for grid node ``(i,j)`` or to NaN if
      no spot is found for this grid node.

    The peak value is that of the spot in the filtered image of the wavefront
    sensor. The row index `i` runs along the direction of `y`, while the column
    index `j` runs along the direction of `x`. Coordinates `x` and `y` are in
    pixels.

    """
    def __init__(self, *, x, y, z, step, origin, **kwds):
        # Check and set mandatory attributes.
        x = np.asarray(x)
        shape = x.shape
        if len(shape) != 2:
            raise ValueError('x, y, and z must be 2-dimensional arrays')
        y = np.asarray(y)
        z = np.asarray(z)
        if not y.shape == z.shape == shape:
            raise ValueError('x, y, and z must have the same shape')
        self.x = x
        self.y = y
        self.z = z
        self.shape = shape
        self.step = step
        self.origin = origin

        # Optional attributes.
        for (key, val) in kwds.items():
            setattr(self, key, val)

    def __repr__(self):
        fix = lambda x: round(x, 3)
        xorg, yorg = map(fix, self.origin)
        step = fix(self.step)
        nrows, ncols = self.shape
        n = self.count_number_of_valid_nodes()
        return (f'ShackHartmannGrid(step={step}, origin=({xorg}, {yorg}), ...)\n' +
                f'with {nrows} row(s), {ncols} column(s), and {n} valid nodes')

    def mask_of_valid_nodes(self):
        """Return the mask of the valid nodes."""
        return np.logical_not(np.isnan(self.z))

    def count_number_of_valid_nodes(self):
        """Return the number of valid nodes."""
        return np.count_nonzero(self.mask_of_valid_nodes())

class ShackHartmannData:
    """Shack-Hartmann data.

    The object is created given Shack-Hartmann data.

    Mandatory attributes (n is the number of measured slopes):

    - `.i`: n-vector of row indices in WFS grid.
    - `.j`: n-vector of column indices in WFS grid.
    - `.x`: n-vector of center of gravity abscissae.
    - `.y`: n-vector of center of gravity ordinates.
    - `.z`: n-vector of spot intensity.
    - `.cxx`: n-vector of variance of `x`.
    - `.cxy`: n-vector of covariance of `x` and `y`.
    - `.cyy`: n-vector of variance of `y`.

    The spot intensity is given by the peak value of the spot in the filtered
    image of the wavefront sensor. The row index `i` runs along the direction
    of `y`, while the column index `j` runs along the direction of `x`.
    Coordinates `x` and `y` are in pixels.

    """
    def __init__(self, *, i, j, x, y, z, cxx, cxy, cyy, **kwds):
        # Mandatory attributes.
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.z = z
        self.cxx = cxx
        self.cxy = cxy
        self.cyy = cyy

        # Optional attributes.
        for (key, val) in kwds.items():
            setattr(self, key, val)

# TODO add fitting the spot FWHM
#
# NOTE Markers can be UTF8 symbol like '$\u20DD$',  '$\u20DE$' '$\u20DF$'.
#
# TODO have a single parameter grid_rtol to tune the tolerance relative
#      to the step which accounts for:
#      - how far from a grid node can be a spot

def detect_grid_of_spots(img, *,
                         # General keywords:
                         centroiding=False,
                         verbosity=1, # 0 or None for quiet / 1 for info / 2 for full
                         # Matched filter (stage 1.a):
                         spot_fwhm=5.0,
                         spot_footprint='spline',
                         robust_to_background=False,
                         background_level=0.0,
                         # Spot detection (stage 1.b):
                         detect_radius=None,
                         detect_level=0.0,
                         # Grid geometry (stage 2):
                         grid_step=32,
                         grid_step_bounds=None, # `None` or `(min, max)`
                         grid_rtol=0.15,
                         use_minimize_scalar=False,
                         distance_atol=0.0,
                         distance_rtol=0.3,
                         grid_shape=None, # `None` or `(nrows, ncols)`
                         # Centers of gravity:
                         centroid_radius=4, # radius of the ROI to compute the centroid
                         centroid_threshold=0.0, # hard thresholding level, must be ≥ 0
                         # Plotting:
                         fig=None, # `None` to not plot anything
                         # - Keywords for plotting the grid:
                         gridcolor='lightgrey', # None to not draw
                         gridlinewidth=0.3,
                         # - Markers used for plotting spots:
                         spotmarker='o', # `None` to not draw,
                         spotmarkersize=13.0, # in pt (1 inch = 72 pt)
                         spotmarkercolor='lightgrey',# None to not draw
                         # - Markers used for plotting centroids:
                         centroidmarker='x',
                         centroidmarkersize=6.0, # in pt (1 inch = 72 pt)
                         centroidmarkercolor='gold'):
    """Detect a Cartesian grid of evenly spaced spots in an image.

    Positional arguments
    --------------------
    The only positional argument is the image `img` acquired by the
    Shack-Hartmann wavefront sensor (SH-WFS). It is assumed that this images
    has been correctly pre-processed to compensate for the bias and non-uniform
    detector response.

    Description
    -----------
    The algorithm has the following stages:

    1. A matched filter (whose shape is driven by keywords `spot_fwhm`,
       `spot_footprint`, and `robust_to_background`) is applied to the image to
       enhance the spots and facilitate their detection. The local maxima (in
       neighborhoods of radius set by keyword `detect_radius`) in the filtered
       image that are brighter than `detect_level` are considered as detected
       spots.

    2. The parameters of a Cartesian grid are adjusted to the detected spots so
       as to minimize the weighted (by the peak value of the spots) mean of the
       squared distance of the detected spots to the nearest grid node. The
       initial grid step is given by the value of the keyword `grid_step`, the
       brightest detected spot is initially considered to sit at a grid node.
       The optimization strategy consists in first fitting the grid step, then
       the grid offsets, and finally all these parameters. Spots that are too
       far from a node (according to the tolerances specified by keywords
       `distance_atol` and `distance_rtol`) are discarded. If the grid of nodes
       matched by one of the remaining spots is larger than the shape set by
       the keyword `grid_shape`, the sub-grid of the requested size that
       contains the brightest spots (in total) is selected.

    3. If keyword `centroiding` is `False` (the default), the algorithm returns
       an instance of `ShackHartmannGrid` representing the grid geometry.

    4. If keyword `centroiding` is `True`, the centers of gravity of the spots
       (and their covariances) are computed and the algorithm returns an
       instance of `ShackHartmannData` representing the measurements.

    Returned result
    ---------------
    The returned result is a structured object whose content depend on keyword
    `centroiding`.

    Keywords
    --------
    General keywords:

    - `centroiding` specifies whether to compute the centers of gravity of the
      spots and affect the type of the result returned by the algorithm. See
      above for a description.

    - `verbosity` specifies whether to be quiet (0 or `None`), to print minimal
      information (1), or more information (2 or more).

    For the filtered image (stage 1.a):

    - `spot_fwhm` specifies an estimate of the full-width at half-maximum of
      the spots. This parameter is used to build the detection filter. If
      specified, ``spot_fwhm > 0` must hold. If not specified, then
      `centroid_radius` must be specified and ``spot_fwhm = centroid_radius``
      is chosen.

    - `spot_footprint` specifies the footprint of the spots, one of `spline`
      (to use a cubic B-spline) or `Gaussian`. This parameter is used to build
      the detection filter.

    - `robust_to_background` specifies whether the spot detection filter
      should be insensitive to a local background (bias plus slope).

    - `background_level` specifies the mean background level. This value is
      used to extrapolate the image when it is correlated with the detection
      filter.

    For detecting the spots (stage 1.b):

    - `detect_radius` specifies the radius of the local neighborhoods for
      finding the local maxima.

    - `detect_level` specifies the detection threshold for the local maxima in
      the filtered image. The default value is 0. A better threshold is a few
      times the standard deviation of the background noise.

    For fitting the grid (stage 2):

    - `grid_step` specifies an initial estimate of the grid step in pixels.
      Unless this parameter is locked (by setting ``grid_step_bounds =
      (grid_step, grid_step)``), this grid step is refined (as are the other
      grid parameters) to best fit the positions of the detected spots.

    - `grid_step_bounds` specifies the bounds for the grid step. If `None`, the
      lower bound is given by ``(1 - grid_rtol)*grid_step`` and the upper
      bound is given by ``(1 + grid_rtol)*grid_step``.

    - `grid_rtol` specifies a relative tolerance for the grid geometry. It must
      be in the open range (0,1). This parameter is used to compute the grid
      step bounds if `grid_step_bounds` is not specified.

    - `use_minimize_scalar` specifies whether to use
      `scipy.optimize.minimize_scalar` to adjust the initial grid step (unless
      this parameter is locked by the bounds specified in `grid_step_bounds`).
      If `False` (the default), scipy.optimize.fminbound` is used instead.

    - `distance_atol` and `distance_rtol` specify an absolute and relative
      tolerances for the maximal distance between a valid spot and the nearest
      grid node. This criterion is applied after fitting the grid parameters.
      The maximal distance is computed from the adjusted grid `step` by
      ``max(distance_atol, distance_rtol*step)``. The distance criterion has no
      incidence on the spot selection if `distance_rtol` is greater than
      ``sqrt(2)``.

    - `grid_shape` specifies the grid shape, as ``(nrows, ncols)``, of the
      grid. If the "maximal grid" (the grid with all the nodes matching a
      detected spot) is larger than `grid_shape`, the maximal grid is truncated
      so as to preserve the brightest spots. If ``grid_shape = None``, the
      maximal grid is returned.

    For the centers of gravity (stage 4):

    - `centroid_radius` specifies the radius of the sliding window used to
      compute the centers of gravity. The width and height of this window are
      both equal to ``2*centroid_radius + 1``. The window is centered at the
      peak of each detected spot. If specified, ``round(centroid_radius) ≥ 1`
      must hold. If not specified, then `spot_fwhm` must be specified and
      ``centroid_radius = max(1, round(spot_fwhm))`` is chosen.

    - `centroid_threshold` specifies the mean background level. The pixels of
      the image `img` whose value is lower of equal this level are set to zero
      for computing the center of gravity. This level shall be non-negative.

    For plotting:

    - `fig` is `None` to plot nothing, otherwise it is a figure number to have
      image being processed shown in figure ``fig``, the filtered image with
      detected spots and grid lines in figure ``fig + 1``, and, if `verbosity ≥
      2`, the image dilation shown in figure ``fig + 2``.

    - `gridcolor` and `gridlinewidth` specify the settings for plotting the
      grid. The grid is not plotted if `fig` or `gridcolor` is `None`, or if
      `gridlinewidth ≤ 0`.

    - `centroidmaker`, `centroidmarkercolor`, and `centroidmarkersize` specify
      the settings for plotting markers at the positions of detected centroids.
      Not markers are plotted if `fig` or `centroidmarker` is `None`.

    - `spotmaker`, `spotmarkercolor`, and `spotmarkersize` specify the settings
      for plotting markers at the positions of detected spots. Not markers are
      plotted if `fig` or `spotmarker` is `None`.

    Usage
    -----

    The algorithm is designed to be robust but its behavior depends on the
    correct settings of a few parameters.

    The most important settings are `grid_step`, `grid_shape`, and `spot_fwhm`
    (and/or `centroid_radius`). These parameters depends on the considered
    wavefront sensor and shall not depend on the observed source or acquisition
    conditions.

    Different possibilities to reduce the number of false positive (in stages 1
    and 2):

    - Increase the radius of the local neighborhood set by `detect_radius`
      to reduce the number of possible local maxima.

    - Increase the value of the detection threshold `detect_level` to discard
      to faint spots. This level may be set to a few times the standard
      deviation of the background noise.

    - Reduces the tolerances `distance_atol` and/or `distance_rtol` to
      discard spots that are too far from a node of the grid.

    """
    # Set default parameters.
    if not 0 < grid_rtol < 1:
        raise ValueError('grid relative tolerance `grid_rtol` must be > 0 and < 1')
    if not grid_step > 0:
        raise ValueError('`grid_step` must be > 0')
    if grid_step_bounds is None:
        grid_step_bounds = ((1.0 - grid_rtol)*grid_step,
                            (1.0 + grid_rtol)*grid_step)
    elif vector_like_length(grid_step_bounds) != 2:
        raise ValueError('`grid_step_bounds` must be a vector like of 2 values')
    grid_step_min, grid_step_max = grid_step_bounds
    if not 2 <= grid_step_min <= grid_step <= grid_step_max:
        raise ValueError('invalid grid step initial value or bounds')
    if detect_radius is None:
        detect_radius = min(round(0.7*grid_step_min), grid_step_min - 1) # TODO grid_rtol
    detect_radius = round(detect_radius)
    if detect_radius < 1:
        raise ValueError('`detect_radius` is too small, must be ≥ 1')
    if detect_radius >= grid_step_min:
        raise ValueError('`detect_radius` is too large or `gris_step_min` too small')
    if verbosity is None:
        verbosity = 0
    elif not isinstance(verbosity, int) or verbosity < 0:
        raise ValueError('`verbosity` must be `None` or an integer ≥ 0')
    if not distance_atol >= 0:
        raise ValueError('`distance_atol` must be ≥ 0')
    if not distance_rtol >= 0:
        raise ValueError('`distance_rtol` must be ≥ 0')

    # By default, centroid_radius and spot_fwhm are the same (except that centroid_radius
    # is rounded to the nearest integer).
    if centroid_radius is None and spot_fwhm is None:
        raise ValueError('at least one of `centroid_radius` or `spot_fwhm` must be specified')
    if centroid_radius is not None and not round(centroid_radius) >= 1:
        raise ValueError('`round(centroid_radius) ≥ 1` does not hold')
    if spot_fwhm is not None and not spot_fwhm > 0:
        raise ValueError('`spot_fwhm > 0` does not hold')
    if spot_fwhm is None:
        spot_fwhm = centroid_radius
    if centroid_radius is None:
        centroid_radius = max(1, round(spot_fwhm))
    else:
        centroid_radius = round(centroid_radius)

    if verbosity > 0:
        fix = lambda x: round(x, 4)

    if fig is not None:
        plt.ion()

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.imshow(img, origin='lower', vmin=0)
        plt.title('Debiased WFS image')
        plt.colorbar(location='right')

    # Build a "detection" filter for the spots. The filter is normalized so
    # that its output is approximately equal to the peak value of the spot (for
    # spots having the same shape as the filter).
    #spot_footprint = spot_footprint.lower() # case does not matter
    if spot_footprint == 'spline':
        spot_filter = cubic_bspline_2d_filter(spot_fwhm)
    elif spot_footprint == 'Gaussian':
        spot_filter = gaussian_2d_filter(spot_fwhm)
    else:
        raise ValueError('`spot_footprint` must be "spline" or "Gaussian"')
    if robust_to_background:
        # Remove the mean of the filter so that the output of the filter is
        # insensitive to a local background (bias + slope).
        spot_filter = spot_filter - spot_filter.mean()
    den = (spot_filter**2).sum()
    if den != 1:
        spot_filter /= den

    # Compute a "detection map" by correlating the filter with the image.
    flt = scipy.ndimage.correlate(img, weights=spot_filter,
                                  mode='constant', cval=background_level)
    if fig is not None:
        plt.figure(fig + 1)
        plt.clf()
        plt.imshow(flt, origin='lower', vmin=0)
        plt.title('Filtered WFS image')
        plt.colorbar(location='right')

    # Find the local maxima of the "detection map".
    lmax = scipy.ndimage.grey_dilation(flt, footprint=ball_2d_window(detect_radius))
    if fig is not None and verbosity > 1:
        plt.figure(fig + 2)
        plt.clf()
        plt.imshow(lmax, origin='lower', vmin=0)
        plt.title('Dilated filtered WFS image')
        plt.colorbar(location='right')

    # Boolean mask true at location of positive local maxima.
    cond = (flt == lmax) & (flt > detect_level)
    if detect_radius > 0:
        # Discard local maxima too close to the image boundaries.
        cond[:detect_radius, :] = False
        cond[:, :detect_radius] = False
        cond[-detect_radius:, :] = False
        cond[:, -detect_radius:] = False

    # Extract characteristics of local maxima: their coordinates (x,y) and their
    # intensity z.
    peaks_y, peaks_x = np.nonzero(cond) # coordinates of peaks
    peaks_z = flt[cond]                 # intensity of peaks
    # perm = np.argsort(peaks_z)[::-1]  # local max. in decreasing intensity order

    # Initial grid parameters.
    kmax = peaks_z.argmax() # index of brightest spot
    xoff = peaks_x[kmax]
    yoff = peaks_y[kmax]
    step = grid_step

    # Grid cost function.
    dist = ChamferDistanceToGrid(peaks_z, peaks_x, peaks_y,
                                 xoff=xoff, yoff=yoff, step=step,
                                 auto_update=True)

    # TODO Refine the grid parameters (once) after applying `max_distance` criterion.

    # Unless grid step is locked, first optimize the grid step.
    if grid_step_min < grid_step_max:
        dist.reset()
        if verbosity > 0:
            print('')
        if use_minimize_scalar:
            res = scipy.optimize.minimize_scalar(dist, (dist.get_step(),),
                                                 bounds=grid_step_bounds,
                                                 options={'maxiter': 50,
                                                          'disp': verbosity > 1})
        else:
            if verbosity < 2:
                disp = 1 # non-convergence notification messages only.
            elif verbosity == 2:
                disp = 2 # print a message on convergence too.
            else:
                disp = 3 # print iteration results
            res = scipy.optimize.fminbound(dist, grid_step_min, grid_step_max,
                                           disp=disp)
        if verbosity > 0:
            print(f'Optimize grid step (in {dist.ncalls} calls):')
            print(f'  Grid offsets: ({fix(xoff)}, {fix(yoff)})')
            print(f'  Grid step: {fix(step)} -> {fix(dist.step)}')
            print(f'  Mean distance: {fix(dist.get_mean_distance())} pixels')
        step = dist.step

    # Optimize on the grid offsets.
    dist.reset()
    if verbosity > 0:
        print('')
    res = scipy.optimize.minimize(dist, dist.get_offsets(), method='Powell',
                                   options={'maxiter': 100,
                                            'disp': verbosity > 1})
    if verbosity > 0:
        print(f'Optimize grid offsets (in {dist.ncalls} calls):')
        print(f'  Grid offsets: ({fix(xoff)}, {fix(yoff)}) -> ({fix(dist.xoff)}, {fix(dist.yoff)})')
        print(f'  Grid step: {fix(step)}')
        print(f'  Mean distance: {fix(dist.get_mean_distance())} pixels')
    xoff, yoff = dist.get_offsets()

    # Optimize on all grid parameters.
    dist.reset()
    if verbosity > 0:
        print('')
    res = scipy.optimize.minimize(dist, (xoff, yoff, step), method='Powell',
                                  options={'maxiter': 100,
                                           'disp': verbosity > 1})
    if verbosity > 0:
        print(f'Optimize all grid parameters (in {dist.ncalls} calls):')
        print(f'  Grid offsets: ({fix(xoff)}, {fix(yoff)}) -> ({fix(dist.xoff)}, {fix(dist.yoff)})')
        print(f'  Grid step: {fix(step)} -> {fix(dist.step)}')
        print(f'  Mean distance: {fix(dist.get_mean_distance())} pixels')
    xoff, yoff = dist.get_offsets()
    step = dist.get_step()

    # For each local max., compute 2-dimensional index of nearest grid node.
    peaks_j = np.round((peaks_x - xoff)/step).astype(int) # column index
    peaks_i = np.round((peaks_y - yoff)/step).astype(int) # row index

    # Reject spots that are too far from their nearest grid node.
    max_distance = max(distance_atol, distance_rtol*step)
    if max_distance < np.sqrt(2)*step:
        cond = ((xoff + peaks_j*step - peaks_x)**2 +
                (yoff + peaks_i*step - peaks_y)**2
                <= max_distance**2)
        n = np.count_nonzero(cond)
        npeaks = len(cond)
        if verbosity > 0:
            print(f'\nApply distance ≤ {fix(max_distance)} pixels criterion:')
            if n < npeaks:
                print(f'  -> {npeaks - n} spot(s) out of {npeaks} are too far from the grid.')
            else:
                print(f'  -> All {npeaks} spot(s) are close enough to the grid.')
        if n < npeaks:
            peaks_i = peaks_i[cond]
            peaks_j = peaks_j[cond]
            peaks_x = peaks_x[cond]
            peaks_y = peaks_y[cond]
            peaks_z = peaks_z[cond]

    # Recenter the offset and indices so that indices start at 0.
    i_min = peaks_i.min()
    j_min = peaks_j.min()
    if i_min != 0:
        yoff += i_min*step
        peaks_i -= i_min
    if j_min != 0:
        xoff += j_min*step
        peaks_j -= j_min
    nrows = peaks_i.max() + 1 # number of rows is number of nodes along y
    ncols = peaks_j.max() + 1 # number of columns is number of nodes along x
    if verbosity > 0:
        print(f'\nFound a {nrows} × {ncols} maximal grid of nodes.')

    # Populate a maximal grid whose nodes are connected to the nearest spot. In
    # case of tie, the brightest spot has priority. TODO We could have a
    # criterion that combines the brightness (the higher the better) and
    # distance (the lower the better).
    if grid_shape is not None:
        req_nrows, req_ncols = grid_shape
        shape = (max(nrows, req_nrows), max(ncols, req_ncols))
    else:
        shape = (nrows, ncols)
    NaN = math.nan
    grid_x = np.full(shape, NaN)
    grid_y = np.full(shape, NaN)
    grid_z = np.full(shape, float(detect_level))
    npeaks = len(peaks_z)
    for k in range(npeaks):
        i = peaks_i[k]
        j = peaks_j[k]
        z = peaks_z[k]
        if z > grid_z[i, j]:
            grid_z[i, j] = z
            grid_x[i, j] = peaks_x[k]
            grid_y[i, j] = peaks_y[k]

    if grid_shape is not None and (grid_shape[0] < nrows or grid_shape[1] < ncols):
        # To reduce the grid size, find the region of interest (ROI) with the
        # required shape and which contains the brightest spots.
        best_i = -1
        best_j = -1
        best_score = -np.Inf
        for i in range(nrows - req_nrows + 1):
            for j in range(ncols - req_ncols + 1):
                roi = grid_z[i : i + req_nrows, j : j + req_ncols]
                score = np.sum(roi, where=roi > detect_level)
                if score > best_score:
                    best_score, best_i, best_j = score, i, j
        if verbosity > 0:
            # FIXME ij convention or xy convention?
            print(f'\nBest {req_nrows} × {req_ncols} ROI at offsets ({best_i}, {best_j}).')
        grid_x = grid_x[best_i : best_i + req_nrows, best_j : best_j + req_ncols]
        grid_y = grid_y[best_i : best_i + req_nrows, best_j : best_j + req_ncols]
        grid_z = grid_z[best_i : best_i + req_nrows, best_j : best_j + req_ncols]
        xoff += best_j*step
        yoff += best_i*step
        nrows = req_nrows
        ncols = req_ncols

    if fig is not None and spotmarker is not None:
        plt.figure(fig + 1)
        #plt.scatter(peaks_x, peaks_y, spotmarkersize**2, marker=marker, color=spotmarkercolor)
        plt.plot(peaks_x, peaks_y,
                 linewidth=0, # to not draw lines
                 markerfacecolor=(0,0,0,0), # transparent color to not fill marker
                 markeredgewidth=0.5,
                 marker=spotmarker,
                 markersize=spotmarkersize,
                 markeredgecolor=spotmarkercolor)

    if fig is not None and gridcolor is not None and gridlinewidth > 0:
        plt.figure(fig + 1)
        xmin, xmax = xoff, xoff + (ncols - 1)*step
        ymin, ymax = yoff, yoff + (nrows - 1)*step
        for i in range(nrows):
            y = yoff + i*step
            plt.plot([xmin, xmax], [y, y], color=gridcolor, linewidth=gridlinewidth)
        for j in range(ncols):
            x = xoff + j*step
            plt.plot([x, x], [ymin, ymax], color=gridcolor, linewidth=gridlinewidth)

    if not centroiding:
        cond = grid_z <= detect_level
        if cond.any():
            grid_x[cond] = NaN
            grid_y[cond] = NaN
            grid_z[cond] = NaN
        return ShackHartmannGrid(x=grid_x, y=grid_y, z=grid_z,
                                 step=step, origin=(xoff,yoff))

    # Compute the center of gravity of all valid spots.
    cond = grid_z > detect_level
    n = np.count_nonzero(cond)
    grid_i, grid_j = np.meshgrid(range(nrows), range(ncols), indexing='ij')
    cog_i = grid_i[cond]
    cog_j = grid_j[cond]
    cog_x = grid_x[cond].astype(float)
    cog_y = grid_y[cond].astype(float)
    cog_z = grid_z[cond]
    cog_cxx = np.empty(n, float)
    cog_cxy = np.empty(n, float)
    cog_cyy = np.empty(n, float)

    # FIXME eliminate spots closer to the image edge than the window used for
    #       the center of gravity
    #

    # Coordinates in the ROI relative to center.
    coords = np.array(range(-centroid_radius, centroid_radius + 1)).astype(float)
    roi_y, roi_x = np.meshgrid(coords, coords, indexing='ij')
    roi_w = ball_2d_window(centroid_radius).astype(float)
    roi = ImageROI(xcen=0, ycen=0, width=len(coords), height=len(coords))
    for k in range(n):
        # Central pixel of the window for the center of gravity.
        x = round(cog_x[k])
        y = round(cog_y[k])

        # Extract the ROI in the image and apply a hard thresholding and a
        # mask.
        roi.center_at(x, y)
        roi_z = roi_w*np.maximum(roi(img), centroid_threshold)

        # Compute the center of gravity in the ROI and register it.
        dx, dy, cxx, cxy, cyy = center_of_gravity(roi_z, roi_x, roi_y, covariance=True)
        cog_x[k] = x + dx
        cog_y[k] = y + dy
        cog_cxx[k] = cxx
        cog_cxy[k] = cxy
        cog_cyy[k] = cyy

    if fig is not None and centroidmarker is not None:
        plt.figure(fig + 1)
        plt.plot(cog_x, cog_y,
                 linewidth=0, # to not draw lines
                 markerfacecolor=(0,0,0,0), # transparent color to not fill marker
                 markeredgewidth=0.7,
                 marker=centroidmarker,
                 markersize=centroidmarkersize,
                 markeredgecolor=centroidmarkercolor)

    return ShackHartmannData(i=cog_i, j=cog_j, x=cog_x, y=cog_y, z=cog_z,
                             cxx=cog_cxx, cxy=cog_cxy, cyy=cog_cyy)
