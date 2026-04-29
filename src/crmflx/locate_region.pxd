"""
C interface declarations

locreg_c is pure C internals with pointers for allowing batch array processing
locreg uses the Cython cpdef convention for wrapper definition, yet this writes output to a tuple.
"""
cdef void locreg_c(
    double xkp,
    double xgsm,
    double ygsm,
    double zgsm,
    double* xtail,
    double* ytail,
    double* ztail,
    int* idloc
)

cpdef (double, double, double, int) locreg(
    double xkp,
    double xgsm,
    double ygsm,
    double zgsm
)