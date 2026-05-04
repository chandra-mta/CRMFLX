"""
C interface declarations
"""
ctypedef void (*FluxMethod)(
    double xkp, double xkpsc,
    double fmean_a, double fmean_b,
    double f95_a, double f95_b,
    double f50_a, double f50_b,
    double fsg_a, double fsg_b,
    double favg_a, double favg_b,
    double* fmean,
    double* f95,
    double* f50,
    double* fsig,
    double* favg
)

cdef struct SectorSpec:
    double x_left, x_right
    double y_bottom, y_top
    double fmean_a, fmean_b
    double f95_a, f95_b
    double f50_a, f50_b
    double fsg_a, fsg_b
    double favg_a, favg_b
    FluxMethod method
    """
    x_left, x_right     --- parameter set of sector limit in x
    y_bottom, y_top     --- parameter set of sector limit in y
    fmean_a, fmean_b    --- parameter set of fmean computation
    f95_a, f95_b        --- parameter set of f95 computation
    f50_a, f50_b        --- parameter set of f50 computation
    fsg_a, fsg_b        --- parameter set of fsig computation
    favg_a, favg_b      --- parameter set of favg computation
    method              --- Set of satistical calculation functions.
    """

cdef SectorSpec[3] SECTORS_SOLAR_WIND
cdef SectorSpec[4] SECTORS_MAGNETOSHEATH
cdef SectorSpec[10] SECTORS_MAGNETOSPHERE

cdef void sect_comp_c(
    double xkp,
    double xkpsc,
    SectorSpec* spec,
    double* scmean,
    double* sc95,
    double* sc50,
    double* scsig,
    double* xcen,
    double* ycen
)