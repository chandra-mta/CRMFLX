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
from cython.cimports.libc.math import sqrt, sin, cos, tan, asin, atan

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

cpdef double bowshk2(
    double bx,
    double by,
    double bz,
    double vx,
    double vy,
    double vz,
    double dennum,
    double swetemp,
    double swptemp,
    double hefrac,
    double swhtemp,
    double xpos,
    double bowang
):
    """
    this routine is designed to give the bow shock radius, at a
    given x, of the bow shock for any solar wind conditions.
    
    this routine has been optimized for the simulation.
    
    inputs  bx      --- the imf b_x [nt]
            by      --- the imf b_y [nt]
            bz      --- the imf b_z [nt]
            vx      --- the imf v_x [km/s]
            vy      --- the imf v_y [km/s]
            vz      --- the imf v_z [km/s]
            dennum  --- the solar wind proton number density [#/cm^3]
            swetemp --- the solar wind electron temperature [k]
            swptemp --- the solar wind proton temperature [k]
            hefrac  --- fraction of solar wind ions which are helium ions
            swhtemp --- the temperature of the helium [k]
            xpos    --- down tail distance cross section is calculated [re]
            bowang  --- angle bow shock radius calculated (rad).
    
    output: bowrad  --- updated cylindrical radius (re).

    """
#
#--- convert the temperature from kelvins to ev
#
    cdef double etemp  = swetemp / 11600.
    cdef double ptemp  = swptemp / 11600.
    cdef double hetemp = swhtemp / 11600.
    cdef double dpr    = 6.2832  / 360.0
    cdef double rade   = 6378.0
    cdef double pi     = 4.0 * atan(1.0)
    cdef double gamma  = 5.0 / 3.0
#
#--- parameters that specify the shape of the base model
#--- of the bow shock.  the model has the form
#--- rho**2 = a*x**2 -b*x + c.
#---
#--- the parameters are for the greenstadt etal. 1990 model
#--- (grl, vol 17, p 753, 1990)
#
    cdef double xl = 22.073117134
    cdef double x0 =  3.493725046
    cdef double xn = 14.422071657
#
#--- here the eccentricity of the base model is adjusted.  see 
#--- the paper for an explanation
#
    cdef double eps = 1.0040
#
#--- calculate new paramaters for cylindrical shock model after
#--- adjustment of eccentricity.
#
#--- calculate parameters of interest
#
    cdef double btotcgs = sqrt(bx * bx + by * by + bz * bz) * 1.e-5    #--- btot in cgs units
    cdef double btot    = btotcgs * 1.e+5                                   #--- btot in mks units
    cdef double vtot    = sqrt(vx * vx + vy * vy + vz * vz) * 1.e+5    #--- vtot in cm/sec
    cdef double vtot1   = sqrt(vx * vx + vy * vy + vz * vz)            #--- vtot in km/sec
#
#--- the following definitions of v_a and c_s are taken from
#--- slavin & holzer jgr dec 1981
#
    cdef double v_a   = btotcgs / sqrt(4.0 * pi * dennum * 1.67e-24 * (1.0 + hefrac * 4.0)) 
    cdef double pres1 = dennum * ((1.0 + hefrac * 2.0) * etemp + (1 + hefrac * hetemp) * ptemp) * 1.602e-12
    cdef double c_s   = sqrt(2.0 * pres1 / (dennum * 1.67e-24 * (1.0 + hefrac * 4.0)))
#
#--- calculate mach numbers
#
#--- m_f is the fast magnetosonic speed for theta_bn = 90 degrees
#
    cdef double m_a = vtot / v_a
    cdef double m_s = vtot / c_s
    cdef double m_f = vtot / sqrt(v_a * v_a + c_s * c_s)
#
#--- this is the modification for the change in the bow shock due
#--- to changing solar wind dynamic pressure
#
#--- average values of number density and solar wind velocity
#
    cdef double xnave =  7.0
    cdef double vave = 430.0
#
#--- fracpres is the fraction by which all length scales in the
#--- bow shock model will change due to the change in the sola
#--- wind dynamic pressure
#
    cdef double fracpres = ((xnave * vave * vave) / (dennum * vtot1 * vtot1))**(1.0/6.0)

    xn1 = xn * fracpres
    x0  = x0 * fracpres
    xl  = xl * fracpres
#
#--- calculate yet again the parameters for the updated model
#
    cdef double a = eps * eps -1
    cdef double b = 2.0 * eps * xl + 2.0 *( eps * eps -1) * x0
    cdef double c = xl * xl + 2.0 * eps * xl * x0 + (eps * eps -1) * x0 * x0
#
#--- calculate shock with correct pressure
#
    cdef double xtemp = a * xpos**2 - b * xpos + c
    cdef double rho2
    if xtemp < 0:
        rho2 = 0
    else:
        rho2 = sqrt(xtemp)
#
#--- modify the bow shock for the change in flaring due to the
#--- change in local magnetosonic mach number
#
#--- first calculate the flaring angle for average solar wind conditions
#
    cdef double ave_ma   = 9.4
    cdef double ave_ms   = 7.2
    cdef double vwin_ave = 430.0*1.e+5
    cdef double va_ave   = vwin_ave / ave_ma
    cdef double vs_ave   = vwin_ave / ave_ms

    cdef double vms      = sqrt(0.5 * ((va_ave**2 + vs_ave**2) \
                + sqrt((va_ave**2  + vs_ave**2)**2 \
                - 4.0 * va_ave**2 * vs_ave**2 *(cos(45 * dpr)**2))))

    ave_mf   = vwin_ave / vms
    cdef double thet2    = asin(1.0 / ave_mf)
#
#--- now calculate the cylindrical radius, and the y and z coordinates,
#--- of the shock for the angle bowang around the tail axis for a given
#--- xpos.  note that bowang is 0 along the positive z axis
#--- (bowang/dpr is the angle about the tail axis in degrees,
#--- rhox is the updated cylindrical radius of the prevailing bow
#--- shock at downtail distance xpos in earth radii.
#
    vms = fast(bx,by,bz,v_a,c_s,vtot,bowang)

    m_f = vtot / vms
    cdef double thet1 = asin(1.0 / m_f)
    cdef double bowrad = rho2 + (xn1 - xpos) * (tan(thet1) - tan(thet2)) #: rhox
    return bowrad