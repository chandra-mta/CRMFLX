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

