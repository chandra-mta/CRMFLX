"""
This module contains the core functions for the CRMFLX simulation, including:
- rot8ang: rotates a 2D vector about a hinge point in the xy-plane
- locate: defines the position of a point at the model magnetopause closest to a given point in space, and the distance between them
- bowshk2: calculates the bow shock radius at a given x for any solar wind conditions
- locreg: determines which phenomenological region the spacecraft is in (solar wind, magnetosheath, or magnetosphere) based on its coordinates)

Note that convenient python conventions for code design will not always be followed in this modules
as such practices do not always align with the most efficient Cython code design.
"""

import sys
import os
import math
import numpy as np
import cython
from _read_model import load_solar_wind_parameters

#
#--- SET THE KP TOLERANCE VARIABLE.  IF THE KP VALUE FOR WHICH OUTPUT
#--- IS DESIRED CHANGES BY MORE THAN THIS VALUE, THE KP SCALING
#--- PARAMETERS ARE RECALCULATED
#
XKPTOL  = 0.3
#
#--- GLOBAL VARIABLES
#
BLENDX1 = -8.0      #--- XGSM (XTAIL) LIMIT FOR *NOT* USING Z-LAYERS (IF XTAIL >= BLENDX1, NO Z-LAYERS USED)
BLENDX2 = -10.0     #--- XGSM (XTAIL) LIMIT FOR *ONLY* USING Z-LAYERS (IF XTAIL >= BLENDX1, NO Z-LAYERS USED)
MAXCELL = 1000      #--- MAXIMUM NUMBER OF NEAR-NEIGHBOR CELLS USED IN CALCULATING THE AVERAGE FLUX AT THIS LOCATION
MAXNSPHVOL = 100000 #--- MAXIMUM NUMBER OF SUB-VOLUME ELEMENTS STORES IN THE STREAMLINE MAPPING SEARCH VOLUME
MAXPNT  = 250000    #--- MAXIMUM NUMBER OF POINTS USED IN DATAFILE
NSAVE   = 10        #--- THE NUMBER OF RANGE SORTED DATA CELLS TO SAVE BEFORE REMOVING THE HIGHEST & LOWEST FLUX VALUES
NUMSEC  = 10        #--- NUMBER OF SPATILA SECTORS: MAX NUSED IN MAGNETOSPHERE, MAGNETOSHEATH, OR SOLAR WIND
PI      = 3.14159265359
XINC    = 1.0       #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: LENGTH OF VOLUME ELEMENT IN X-DIRECTION (RE)
YINC    = 1.0       #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: LENGTH OF VOLUME ELEMENT IN Y-DIRECTION (RE)
ZINC    = 1.0       #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: LENGTH OF VOLUME ELEMENT IN Z-DIRECTION (RE
XMIN    = -30.0     #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: MINIMUM X-VALUE OF VOLUME ELEMENT (RE)
YMIN    = -30.0     #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: MINIMUM Y-VALUE OF VOLUME ELEMENT (RE)
ZMIN    = -30.0     #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: MINIMUM Z-VALUE OF VOLUME ELEMENT (RE)
XMAX    = +30.0     #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: MAXIMUM X-VALUE OF VOLUME ELEMENT (RE)
YMAX    = +30.0     #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: MAXIMUM Y-VALUE OF VOLUME ELEMENT (RE)
ZMAX    = +30.0     #--- CHANDRA VOLUME ELEMENT DATABASE PARAMETERS: MAXIMUM Z-VALUE OF VOLUME ELEMENT (RE)
NUM2    = 6         #--- SUB-VOLUME ELEMENT DATABASE PARAMETERS: NUMBER OF SUB-VOLUME ELEMENTS IN EACH DIRECTION
XINC2   = 0.1666666 #--- SUB-VOLUME ELEMENT DATABASE PARAMETERS: LENGTH OF SUB-VOLUME ELEMENT IN X-DIRECTION (RE)
YINC2   = 0.1666666 #--- SUB-VOLUME ELEMENT DATABASE PARAMETERS: LENGTH OF SUB-VOLUME ELEMENT IN Y-DIRECTION (RE)
ZINC2   = 1.0       #--- SUB-VOLUME ELEMENT DATABASE PARAMETERS: LENGTH OF SUB-VOLUME ELEMENT IN Z-DIRECTION (RE)

_SOLAR_WIND_PARAMS = load_solar_wind_parameters()

def rot8ang(ang: cython.double, x: cython.double, y: cython.double, xhinge: cython.double):
    """
    rotates the 2-d vector about its hinge point in the xy-plane
    input:  ang    --- angle to rotate (rad).
            x      --- initial x value.
            y      --- initial y value.
            xhinge --- x value of aberration hinge point.
    
    output: xrot2  --- final x value.
            yrot2  --- final y value.
    """
    xrot2: cython.double
    yrot2: cython.double

    if x <= xhinge:
        xrot2 =  x * math.cos(ang) + y * math.sin(ang)
        yrot2 = -x * math.sin(ang) + y * math.cos(ang)
    else:
        xrot2 = x
        yrot2 = y

    return [xrot2, yrot2]

def locate(xn_pd: cython.double, vel: cython.double, xgsm: cython.double, ygsm: cython.double, zgsm: cython.double):
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
    pd: cython.double
    rat: cython.double
    rat16: cython.double
    a: cython.double
    s0: cython.double
    x0: cython.double
    xm: cython.double
    phi: cython.double
    rho: cython.double
    xmgnp: cython.double
    rhomgnp: cython.double
    ymgnp: cython.double
    zmgnp: cython.double
    xid: cython.int
    xksi: cython.double
    xdzt: cython.double
    sq1: cython.double
    sq2: cython.double
    sigma: cython.double
    tau: cython.double
    dist: cython.double
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
        phi = math.atan2(ygsm,zgsm)
    else:
        phi = 0.0

    rho = math.sqrt(ygsm**2 + zgsm**2)

    if xgsm < xm:
#
#--- calculate (x,y,z) for the closest point at the magnetopause
#
        xmgnp   = xgsm
        rhomgnp = a * math.sqrt(s0**2 - 1)
        ymgnp   = rhomgnp * math.sin(phi)
        zmgnp   = rhomgnp * math.cos(phi)
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
        sq1   = math.sqrt((1.0 + xksi)**2 + xdzt**2)
        sq2   = math.sqrt((1.0 - xksi)**2 + xdzt**2)
        sigma = 0.5 * (sq1 + sq2)
        tau   = 0.5 * (sq1 - sq2)
#
#--- calculate (x,y,z) for the closest point at the magnetopause
#
        xmgnp   = x0 - a * (1.0 - s0 * tau)
        rhomgnp = a * math.sqrt((s0**2 - 1.0)*(1.0 - tau**2))
        ymgnp   = rhomgnp * math.sin(phi)
        zmgnp   = rhomgnp * math.cos(phi)
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
    dist = math.dist([xgsm, ygsm, zgsm], [xmgnp, ymgnp, zmgnp])

    return xmgnp, ymgnp, zmgnp, dist, xid

def bowshk2(bx: cython.double,
            by: cython.double,
            bz: cython.double,
            vx: cython.double,
            vy: cython.double,
            vz: cython.double,
            dennum: cython.double,
            swetemp: cython.double,
            swptemp: cython.double,
            hefrac: cython.double,
            swhtemp: cython.double,
            xpos: cython.double,
            bowang: cython.double):
    """
    this routine is designed to give the bow shock radius, at a
    given x, of the bow shock for any solar wind conditions.
    
    references:
    this routine is adpated from the paper by l. bennet et.al.,
    "a model of the earth's distant bow shock."  this paper was
    to be published in the journal of geophysical research, 1997.
    https://ui.adsabs.harvard.edu/abs/1997JGR...10226927B
    their source code was obtained from their web site at:
    http://www.igpp.ucla.edu/galileo/newmodel.htm
    
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
    rho2: cython.double
#
#--- convert the temperature from kelvins to ev
#
    etemp: cython.double  = swetemp / 11600.
    ptemp: cython.double  = swptemp / 11600.
    hetemp: cython.double = swhtemp / 11600.
    dpr: cython.double    = 6.2832  / 360.0
    pi: cython.double     = 4.0 * math.atan(1.0)
#
#--- parameters that specify the shape of the base model
#--- of the bow shock.  the model has the form
#--- rho**2 = a*x**2 -b*x + c.
#---
#--- the parameters are for the greenstadt etal. 1990 model
#--- (grl, vol 17, p 753, 1990)
#
    xl: cython.double = 22.073117134
    x0: cython.double =  3.493725046
    xn: cython.double = 14.422071657
#
#--- here the eccentricity of the base model is adjusted.  see 
#--- the paper for an explanation
#
    eps: cython.double = 1.0040
#
#--- calculate new paramaters for cylindrical shock model after
#--- adjustment of eccentricity.
#
#--- calculate parameters of interest
#
    btotcgs: cython.double = math.sqrt(bx * bx + by * by + bz * bz) * 1.e-5    #--- btot in cgs units
    vtot: cython.double    = math.sqrt(vx * vx + vy * vy + vz * vz) * 1.e+5    #--- vtot in cm/sec
    vtot1: cython.double   = math.sqrt(vx * vx + vy * vy + vz * vz)            #--- vtot in km/sec
#
#--- the following definitions of v_a and c_s are taken from
#--- slavin & holzer jgr dec 1981
#
    v_a: cython.double = btotcgs / math.sqrt(4.0 * pi * dennum * 1.67e-24 * (1.0 + hefrac * 4.0)) 
    pres1: cython.double = dennum * ((1.0 + hefrac * 2.0) * etemp + (1 + hefrac * hetemp) * ptemp) * 1.602e-12
    c_s: cython.double   = math.sqrt(2.0 * pres1 / (dennum * 1.67e-24 * (1.0 + hefrac * 4.0)))
#
#--- calculate mach numbers
#
#--- m_f is the fast magnetosonic speed for theta_bn = 90 degrees
#
    m_f: cython.double = vtot / math.sqrt(v_a * v_a + c_s * c_s)
#
#--- this is the modification for the change in the bow shock due
#--- to changing solar wind dynamic pressure
#
#--- average values of number density and solar wind velocity
#
    xnave: cython.double =  7.0
    vave: cython.double = 430.0
#
#--- fracpres is the fraction by which all length scales in the
#--- bow shock model will change due to the change in the sola
#--- wind dynamic pressure
#
    fracpres: cython.double = ((xnave * vave * vave) / (dennum * vtot1 * vtot1))**(1.0/6.0)

    xn1: cython.double = xn * fracpres
    x0: cython.double  = x0 * fracpres
    xl: cython.double  = xl * fracpres
#
#--- calculate yet again the parameters for the updated model
#
    a: cython.double = eps * eps -1
    b: cython.double = 2.0 * eps * xl + 2.0 *( eps * eps -1) * x0
    c: cython.double = xl * xl + 2.0 * eps * xl * x0 + (eps * eps -1) * x0 * x0
#
#--- calculate shock with correct pressure
#
    xtemp: cython.double = a * xpos**2 - b * xpos + c
    if xtemp < 0:
        rho2 = 0
    else:
        rho2 = math.sqrt(xtemp)
#
#--- modify the bow shock for the change in flaring due to the
#--- change in local magnetosonic mach number
#
#--- first calculate the flaring angle for average solar wind conditions
#
    ave_ma: cython.double   = 9.4
    ave_ms: cython.double   = 7.2
    vwin_ave: cython.double = 430.0*1.e+5
    va_ave: cython.double   = vwin_ave / ave_ma
    vs_ave: cython.double   = vwin_ave / ave_ms

    vms: cython.double      = math.sqrt(0.5 * ((va_ave**2 + vs_ave**2) \
                + math.sqrt((va_ave**2  + vs_ave**2)**2 \
                - 4.0 * va_ave**2 * vs_ave**2 *(math.cos(45 * dpr)**2))))

    ave_mf: cython.double   = vwin_ave / vms
    thet2: cython.double    = math.asin(1.0 / ave_mf)
#
#--- now calculate the cylindrical radius, and the y and z coordinates,
#--- of the shock for the angle bowang around the tail axis for a given
#--- xpos.  note that bowang is 0 along the positive z axis
#--- (bowang/dpr is the angle about the tail axis in degrees,
#--- rhox is the updated cylindrical radius of the prevailing bow
#--- shock at downtail distance xpos in earth radii.
#
    vms      = fast(bx,by,bz,v_a,c_s,vtot,bowang)
    m_f: cython.double      = vtot / vms
    thet1: cython.double    = math.asin(1.0 / m_f)
    xtemp1: cython.double   = xn1 - xpos
    rhox: cython.double     = rho2 + xtemp1 * (math.tan(thet1) - math.tan(thet2))
    bowrad: cython.double   = rhox

    return bowrad

def fast(bx: cython.double,
         by: cython.double,
         bz: cython.double,
         va: cython.double,
         vs: cython.double,
         v0: cython.double,
         alp: cython.double):
    """
    local fast magnetosonic speed

    it uses the simple bisection method to solve the equations.
    accuracy to 0.01 in v_ms is good enough 
    
        this routine is adapted from the paper by l. bennet et.al.,
        "a model of the earth's distant bow shock."  this paper was
        to be published in the journal of geophysical research, 1997.
        this source code is from their web site at:
        http://www.igpp.ucla.edu/galileo/newmodel.htm
    """
    btot: cython.double = math.sqrt(bx * bx + by * by + bz * bz) #: Total interplanetary magnetic field strength.
    vx: cython.double   = 1
    va1: cython.double  = va / 1.0e5
    vs1: cython.double  = vs / 1.0e5
    v01: cython.double  = v0 / 1.0e5

    func: cython.double = fast_func(bx, by, bz, alp, btot, vx, va1, vs1, v01)


    step1: cython.double = 2.0
    step2: cython.double = 0.1
    step3: cython.double = 0.01
    
    #: Step size and sign is the same regardless of sign of initial
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

    vy    = math.sin(alp) * math.sqrt(v01 * vx - vx**2)
    vz    = math.cos(alp) * math.sqrt(v01 * vx - vx**2)
    vms   = 1.0e5 * math.sqrt(vx * vx + vy * vy + vz * vz)

    return vms

def fast_func( bx,  by,  bz,  alp, btot,  vx,  va1,  vs1,  v01):

    out  = va1**2 + vs1**2 - 2.0 * v01 * vx + math.sqrt((va1**2 + vs1**2)**2         \
           - 4.0 *va1**2 * vs1**2 * (vx * bx + by * math.sin(alp) * math.sqrt(v01*vx \
           - vx**2) + bz * math.cos(alp) * math.sqrt(v01*vx                          \
           - vx**2))**2 / (btot**2 *v01 * vx))

    return out

