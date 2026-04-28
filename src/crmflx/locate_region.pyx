"""
this module determines which phenomenological region the spacecraft is in,
as well as the geographical coordinates of the spacecraft in the geotail system.
"""
from .bowshock cimport bowshk2

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

cdef (double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double) solwind(double xkp):
    """
    get the solar wind parameters used as inputs for the bow shock
    and magnetopause boundary models. Returns 15 values.
    
    input:  xkp     --- kp index (real value between 0 & 9).
    output: bx      --- the imf b_x [nt]
            by      --- the imf b_y [nt]
            bz      --- the imf b_z [nt]
            vx      --- x component of solar wind bulk flow velocity (km/s).
            vy      --- y component of solar wind bulk flow velocity (km/s).
            vz      --- z component of solar wind bulk flow velocity (km/s).
            dennum  --- the solar wind proton number density [#/cm^3]
            swetemp --- the solar wind electron temperature [k]
            swptemp --- the solar wind proton temperature [k]
            hefrac  --- fraction of solar wind ions which are helium ions
            swhtemp --- the temperature of the helium [k]
            bowang  --- angle bow shock radius calculated (rad). Instantiated as PI with possbile refinement later if necessary.
            dypres  --- solar wind dynamic pressure (np).
            abang   --- aberration angle of magnetotail (deg).
            xhinge  --- hinge point of magnetotail (re).
    """

    cdef double bx      =   -5.0
    cdef double by      =    6.0
    cdef double bz      =    6.0
    cdef double vx      = -500.0
    cdef double vy      =    0.0
    cdef double vz      =    0.0

    cdef double dennum  = 8.0
    cdef double swetemp = 1.4e+5
    cdef double swptemp = 1.2e+5
    cdef double hefrac  = 0.047
    cdef double swhtemp = 5.8e+5
    cdef double bowang  = 3.14159265359
    cdef double dypres
    cdef double abang
    cdef double xhinge

    if xkp <= 4.0:
        dypres  =    1.0
        abang   =    4.0
        xhinge  =   14.0
        vx      = -400.0

    elif (xkp > 4.0) and (xkp <= 6.0): 
        dypres1 =    1.0
        abang1  =    4.0
        vx1     = -400.0
        dypres2 =    4.0
        abang2  =    0.0
        vx2     = -500.0

        dypres  = y_interpolate(4.0, dypres1, 6.0, dypres2, xkp)
        abang   = y_interpolate(4.0, abang1,  6.0, abang2,  xkp)
        xhinge  = 14.0
        vx      = y_interpolate(4.0, vx1,     6.0, vx2,     xkp)

    else:
        dypres1 =    4.0
        dypres2 =   10.0
        dypres  = y_interpolate(6.0, dypres1, 9.0, dypres2, xkp)
        abang   =    0.0
        xhinge  =   14.0
        vx      = -500.0

    return bx, by, bz, vx, vy, vz, dennum, swetemp, swptemp, hefrac, swhtemp, bowang, dypres, abang, xhinge