# -*- coding: utf-8 -*-

"""Linear conjugate gradient."""

from .utils import inner
import numpy as np
import math

def conjgrad(A, b, x=None, inplace=False, maxiter=50, gatol=0.0, grtol=1e-5,
             quiet=False, viewer=None, restart=50):
    """
    Solve the linear problem `A⋅x = b` by the conjugate gradient method.

    Argument `A` is a function to compute the product of the left hand-side
    (LHS) "matrix" of the problem with a vector given in argument.

    Argument `b` is the right hand-side (RHS) "vector".

    Optional argument `x0` is an initial solution to start with. By default,
    the initial solution is zero everywhere. If keyword `inplace` is `True`,
    the returned solution and the initial solution share the same array.

    Optional arguments `gatol` and `grtol` (0.0 and 1E-5 respectively by
    default) are the absolute and relative gradient tolerance for convergence.
    The conjugate gradient iterations are performed until the squared Euclidean
    norm of the residual is less or equal the largest between `gatol` and
    `grtol` times the squared Euclidean norm of the initial residuals.

    Optional argument `maxiter` (50 by default) is the maximum number of
    iterations to perform until returning the result. A warning message is
    printed if the number of iterations is exceeded (unless `quiet` is true).

    Optional argument `restart` (50 by default) is the maximum number of
    successive iterations to perform before restarting the conjugate gradient
    recurrence. Ignored if `restart < 1`.

    """

    # Compute initial residuals.
    if x is None:
        r = np.copy(b)
        x = np.zeros(b.shape, dtype=b.dtype)
    else:
        if not inplace:
            x = np.copy(x)
        r = b - A(x)
    rho = inner(r, r)
    epsilon = max(gatol, grtol*math.sqrt(rho))

    # Conjugate gradient iterations.
    beta = 0.0
    k = 0 # iteration number
    while True:
        if viewer is not None:
            viewer(x, k, math.sqrt(rho))
        if math.sqrt(rho) <= epsilon:
            break
        k += 1
        if k > maxiter:
            if not quiet:
                print("WARNING: too many iterations ({0})".format(maxiter))
            break

        # Next search direction.
        if beta == 0.0 or (restart > 0 and ((k - 1) % restart) == 0):
            p = r
        else:
            p = r + beta*p

        # Make optimal step along search direction.
        q = A(p)
        gamma = inner(p, q)
        if gamma <= 0.0:
            raise ValueError("Operator A is not positive definite")
        alpha = rho/gamma
        x += alpha*p
        r -= alpha*q
        rho_prev, rho = rho, inner(r, r)
        beta = rho/rho_prev
    return x
