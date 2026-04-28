"""
C interface declarations
"""
cdef double bowshk2_c(
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
)
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
)