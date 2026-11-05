# -*- coding: utf-8 -*-

from .fits import readfits, writefits

from .centroids import center_of_gravity, fix_weights, invert_2x2_symmetric_matrix

from .metrics import ChamferDistanceToGrid
from .metrics import offset_to_nearest_grid_node, index_of_nearest_grid_node

from .filters import ball_2d_window
from .filters import GAUSSIAN_FWHM, gaussian_1d_filter, gaussian_2d_filter
from .filters import CUBIC_BSPLINE_FWHM, cubic_bspline, cubic_bspline_1d_filter, cubic_bspline_2d_filter

from .images import ImageROI

from .onlinestats import SampleStatistics

from .optim import conjgrad

from .utils import include
from .utils import rand, randn
from .utils import fftshift, ifftshift, fft, ifft
from .utils import abs2, normalize, zeropad
from .utils import DtD
from .utils import outer_sum, outer_diff, outer_prod
from .utils import Result
from .utils import is_vector_like, vector_like_length

from .shack_hartmann import detect_grid_of_spots

from numpy import conj

#from numpy.random import random_sample, standard_normal
