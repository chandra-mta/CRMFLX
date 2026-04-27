"""
this module is designed to give the bow shock radius, at a
given x, of the bow shock for any solar wind conditions.


references:
this routine is adpated from the paper by l. bennet et.al.,
"a model of the earth's distant bow shock."  this paper was
to be published in the journal of geophysical research, 1997.
their source code was obtained from their web site at:
http://www.igpp.ucla.edu/galileo/newmodel.htm

Source paper: https://ui.adsabs.harvard.edu/abs/1997JGR...10226927B

For ideal Cython implementations, certain design choices from the following tutorials have been made.
- https://cython.readthedocs.io/en/latest/src/tutorial/external.html#id5
- 

"""
from cython.cimports.libc.math import sqrt, sin, cos


cdef inline double fast_func(double bx,
                             double by,
                             double bz,
                             double alp,
                             double btot,
                             double vx,
                             double va1,
                             double vs1,
                             double v01):

    """
    Function product for use in the bisectional method of finding the root of the fast magnetosonic speed function. EQ5 in paper.
    """
    cdef double out
    
    out  = va1**2 + vs1**2 - 2.0 * v01 * vx + sqrt((va1**2 + vs1**2)**2         \
           - 4.0 *va1**2 * vs1**2 * (vx * bx + by * sin(alp) * sqrt(v01*vx \
           - vx**2) + bz * cos(alp) * sqrt(v01*vx                          \
           - vx**2))**2 / (btot**2 *v01 * vx))

    return out