"""
Numerical operations shared by multiple modules.
"""
from cython.cimports.libc.math import sqrt, cos, sin

cdef inline double y_interpolate(
    double x1,
    double y1,
    double x2,
    double y2,
    double xin
):
    """
    return y coordinate corresponding to xin along the line define by (x1, y1)/(x2, y2)
    input:  (x1, y1)/(x2, y2)   --- coordinates of two points
            xin                 --- x value to be estimated
    output: yout                --- resulted y value
    """
    cdef double yout = y1 - (y1 - y2) *(x1 -xin) / (x1 - x2)
    return yout

cdef double compute_rng(double xve, double yve, double zve, double xgsm, double ygsm, double zgsm):
    """
    Compute the distance between the spacecraft and the Earth center using euclidean distance.
    """
    cdef double rng

    rng = sqrt((xve - xgsm)**2 + (yve - ygsm)**2 + (zve - zgsm)**2)

    return rng

cdef (double, double) rot8ang(double ang, double x, double y, double xhinge):
    """
    rotates the 2-d vector about its hinge point in the xy-plane
    input:  ang    --- angle to rotate (rad).
            x      --- initial x value.
            y      --- initial y value.
            xhinge --- x value of aberration hinge point.
    
    output: xrot2  --- final x value.
            yrot2  --- final y value.
    """
    cdef double xrot2
    cdef double yrot2

    if x <= xhinge:
        xrot2 =  x * cos(ang) + y * sin(ang)
        yrot2 = -x * sin(ang) + y * cos(ang)
    else:
        xrot2 = x
        yrot2 = y

    return xrot2, yrot2