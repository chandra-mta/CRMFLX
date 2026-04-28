"""
this module determines which phenomenological region the spacecraft is in,
as well as the geographical coordinates of the spacecraft in the geotail system.
"""
from .bowshock cimport bowshk2
from .numerical cimport y_interpolate, compute_rng, rot8ang
from cython.cimports.libc.math import sqrt, sin, cos, atan2

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

def locate(
    double xn_pd,
    double vel,
    double xgsm,
    double ygsm,
    double zgsm
):
    """
    this subroutine defines the position of a point (xmgnp,ymgnp,zmgnp)
    at the model magnetopause, closest to a given point of space
    (xgsm,ygsm,zgsm),   and the distance between them (dist)
    
     nput:  xn_pd --- either solar wind proton number density (per c.c.) (if vel>0)
                      or the solar wind ram pressure in nanopascals   (if vel<0)
            vel   --- either solar wind velocity (km/sec)
                      or any negative number, which indicates that xn_pd stands
                      for the solar wind pressure, rather than for the density
    
            xgsm,ygsm,zgsm - coordinates of the observation point in earth radii
    
    output: xmgnp,ymgnp,zmgnp - coordinates of a point at the magnetopause,
                                closest to the point  xgsm,ygsm,zgsm
            dist  ---  the distance between the above two points, in re,
            xid   ---  indicator; id=+1 and id=-1 mean that the point
                       (xgsm,ygsm,zgsm)  lies inside or outside
                       the model magnetopause, respectively
    
    the pressure-dependent magnetopause is that used in the t96_01 model
    coded by:  n.a. tsyganenko, aug.1, 1995;  revised  june 22, 1996

    """
    cdef double pd
    cdef double rat
    cdef double rat16
    cdef double a
    cdef double s0
    cdef double x0
    cdef double xm
    cdef double phi
    cdef double rho
    cdef double xmgnp
    cdef double rhomgnp
    cdef double ymgnp
    cdef double zmgnp
    cdef int xid
    cdef double xksi
    cdef double xdzt
    cdef double sq1
    cdef double sq2
    cdef double sigma
    cdef double tau
    cdef double dist
#
#--- pd is the solar wind dynamic pressure (in nanopascals)
#
    if vel < 0.0:
        pd = xn_pd
    else:
        pd = 1.94e-6 * xn_pd * vel**2
#
#--- ratio of pd to the average pressure, assumed as 2 npa
#
    rat   = pd / 2.0
#
#--- the power in the scaling factor is the best-fit value
#--- obtained from data in the t96_01 version of the model
#
    rat16 = rat**0.14
#
#--- values of the magnetopause parameters for  pd = 2 npa are:
#---   a0    = 70.00 /   s00   =  1.08 /   x00   =  5.48
#--- values of the magnetopause parameters, scaled to the actual pressure
#
    a     = 70.0 / rat16
    s0    = 1.08
    x0    = 5.48 / rat16
#
#--- this is the x-coordinate of the "seam" between the ellipsoid and the cylinder
#
    xm    = x0 - a
#
#--- (for details on the ellipsoidal coordinates, see the paper:
#--- n.a.tsyganenko, solution of chapman-ferraro problem for an
#--- ellipsoidal magnetopause, planet.space sci., v.37, p.1037, 1989)
#
    if (ygsm != 0.0) or (zgsm != 0.0):
        phi = atan2(ygsm,zgsm)
    else:
        phi = 0.0

    rho = sqrt(ygsm**2 + zgsm**2)

    if xgsm < xm:
#
#--- calculate (x,y,z) for the closest point at the magnetopause
#
        xmgnp   = xgsm
        rhomgnp = a * sqrt(s0**2 - 1)
        ymgnp   = rhomgnp * sin(phi)
        zmgnp   = rhomgnp * cos(phi)
#
#--- xid=-1 means that the point lies outside the magnetosphere
#--- xid=+1 means that the point lies inside  the magnetosphere
#
        xid     = 0
        if rhomgnp > rho:
            xid =  1
        elif rhomgnp < rho:
            xid = -1

    else:
        xksi  = (xgsm - x0) / a + 1.0
        xdzt  = rho / a
        sq1   = sqrt((1.0 + xksi)**2 + xdzt**2)
        sq2   = sqrt((1.0 - xksi)**2 + xdzt**2)
        sigma = 0.5 * (sq1 + sq2)
        tau   = 0.5 * (sq1 - sq2)
#
#--- calculate (x,y,z) for the closest point at the magnetopause
#
        xmgnp   = x0 - a * (1.0 - s0 * tau)
        rhomgnp = a * sqrt((s0**2 - 1.0)*(1.0 - tau**2))
        ymgnp   = rhomgnp * sin(phi)
        zmgnp   = rhomgnp * cos(phi)
#
#--- xid=-1 means that the point lies outside the magnetosphere
#--- xid=+1 means that the point lies inside  the magnetosphere
#
        xid = 0
        if sigma > s0:
            xid = -1
        elif sigma < s0:
            xid =  1
#
#--- calculate the eudclidean distance between the point xgsm,ygsm,zgsm and the magnetopause
#
    dist = compute_rng([xgsm, ygsm, zgsm], [xmgnp, ymgnp, zmgnp])

    return xmgnp, ymgnp, zmgnp, dist, xid