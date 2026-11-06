# -*- coding: utf-8 -*-

"""Linear algebra tools."""

def invert_2x2_matrix(a11, a12,
                      a21, a22):
    """Invert a general 2×2 matrix.

    Return a 4-tuple ``(b11, b12, b21, b22)`` such that
    ``B = [[b11, b12], [b21, b22]]`` is the inverse of
    ``A = [[a11, a12], [a21, a22]]``.

    """
    det = a11*a22 - a12*a21
    return (a22/det, -a12/det,
            -a21/det, a11/det)

def invert_2x2_symmetric_matrix(a11,
                                a21, a22):
    """Invert a symmetric 2×2 matrix.

    Return a 3-tuple ``(b11, b21, b22)`` such that
    ``B = [[b11, b21], [b21, b22]]`` is the inverse of
    ``A = [[a11, a21], [a21, a22]]``.

    """
    det = a11*a22 - a21*a21
    return (a22/det,
            -a21/det, a11/det)

def invert_3x3_matrix(a11, a12, a13,
                      a21, a22, a23,
                      a31, a32, a33):
    """Invert a general 3×3 matrix."""
    det = (a11*(a22*a33 - a23*a32) +
           a12*(a23*a31 - a21*a33) +
           a13*(a21*a32 - a22*a31))
    return (
        (a22*a33 - a23*a32)/det, (a13*a32 - a12*a33)/det, (a12*a23 - a13*a22)/det,
        (a23*a31 - a21*a33)/det, (a11*a33 - a13*a31)/det, (a13*a21 - a11*a23)/det,
        (a21*a32 - a22*a31)/det, (a12*a31 - a11*a32)/det, (a11*a22 - a12*a21)/det)

def invert_3x3_symmetric_matrix(a11,
                                a21, a22,
                                a31, a32, a33):
    """Invert a symmetric 3×3 matrix."""
    det = (a11*(a22*a33 - a32**2) +
           a21*(a31*a32 - a21*a33) +
           a31*(a21*a32 - a22*a31))
    return (
        (a22*a33 - a32**2)/det,
        (a31*a32 - a21*a33)/det, (a11*a33 - a31**2)/det,
        (a21*a32 - a22*a31)/det, (a21*a31 - a11*a32)/det, (a11*a22 - a21**2)/det)
