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
from cython.cimports.libc.math import sqrt, sin, cos, acos

cdef inline double fast_func(
    double bx,
    double by,
    double bz,
    double alp,
    double btot,
    double vx,
    double va1,
    double vs1,
    double v01
):
    """
    Function product for use in the bisectional method of finding the root
    of the fast magnetosonic speed function. EQ5 in paper.
    """
    cdef double out
    cdef double diff = v01*vx - vx*vx
    cdef double alfven_sqrd = va1*va1
    cdef double sound_sqrd = vs1*vs1

    out = (
        alfven_sqrd + sound_sqrd
        - 2.0 * v01 * vx
        + sqrt(
            (alfven_sqrd + sound_sqrd)**2
            - 4.0 * alfven_sqrd * sound_sqrd
              * (
                  vx * bx
                  + by * sin(alp) * sqrt(diff)
                  + bz * cos(alp) * sqrt(diff)
                )**2
              / (btot*btot * v01 * vx)
        )
    )

    return out

cdef double fast(
    double bx,
    double by,
    double bz,
    double va,
    double vs,
    double v0,
    double alp
):
    """
    local fast magnetosonic speed. EQ5 in paper.

    it uses the simple bisection method to solve the equations.
    accuracy to 0.01 in v_ms is good enough.

    cdef double angle = acos((bx * vx + by * vy + bz * vz)/(sqrt(bx * bx + by * by + bz * bz)*sqrt(vx * vx + vy * vy + vz * vz)))
    """
    cdef double dpr  = 6.2832/360.0
    cdef double btot = sqrt(bx * bx + by * by + bz * bz)
    cdef double vx   = 1
    cdef double va1  = va / 1.0e5
    cdef double vs1  = vs / 1.0e5
    cdef double v01  = v0 / 1.0e5

    cdef double func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)

    cdef double step1 = 2.0
    cdef double step2 = 0.1
    cdef double step3 = 0.01

    if func > 0:
        while True:
            vx += step1
            func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)
            if func <= 0:
                break

        while True:
            vx -= step2
            func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)
            if func >= 0:
                break

        while True:
            vx += step3
            func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)
            if func <= 0:
                break

    elif func < 0:
        while True:
            vx += step1
            func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)
            if func >= 0:
                break

        while True:
            vx -= step2
            func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)
            if func <= 0:
                break

        while True:
            vx += step3
            func = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)
            if func >= 0:
                break

    cdef double vy = sin(alp) * sqrt(v01 * vx - vx**2)
    cdef double vz = cos(alp) * sqrt(v01 * vx - vx**2)

    cdef double vms = 1.0e5 * sqrt(vx * vx + vy * vy + vz * vz)

    return vms