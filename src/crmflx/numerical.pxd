"""
C interface declarations
"""
cdef inline double y_interpolate(
    double x1,
    double y1,
    double x2,
    double y2,
    double xin
)
cdef double compute_rng(double xve, double yve, double zve, double xgsm, double ygsm, double zgsm)
cdef (double, double) rot8ang(double ang, double x, double y, double xhinge)

cdef inline double lin(kp,a,b)
cdef inline double exp_lin(kp,a,b)
cdef inline double mult_exp(kp,a,b)
cdef inline double cent(left,right)