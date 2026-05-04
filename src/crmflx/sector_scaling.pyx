"""
This module determines the KP scaling factors depending on the spacecraft
location the magnetotail/geotail coordinate system.

All sectors scaling factors calculations numerically mimic the original fortran script.
For the model to fit, we use a set of preconfigured algorithmic choices for calculating flux statistics.
These choices are encoded in C structs which define numerical coefficients as well as numerical functions
on a per statistic basis.
This means that each sector in our model will have a corresponding C struct for that sector's specific scaling

These scaling factors are then stored in data arrays to calculate the flux for a given point.
Note that the function definition paradigm uses C pointers for multiple return values.
"""
from .numerical cimport lin, lin_exp, mult_exp, cent

#: Basic Structures
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

#: Different flux calculation methods
cdef void flux_mtd_1(
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
):
    """
    Method 1: Linear scaling for all statistics.
    """
    fmean[0] = lin(xkp, fmean_a, fmean_b)
    f95[0]   = lin(xkp, f95_a, f95_b)
    f50[0]   = lin(xkp, f50_a, f50_b)
    fsig[0]  = lin(xkp, fsg_a, fsg_b)
    favg[0]  = lin(xkpsc, favg_a, favg_b)

cdef void flux_mtd_2(
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
):
    """
    Method 2: Multiplicative exponential scaling for all statistics.
    """
    fmean[0] = mult_exp(xkp, fmean_a, fmean_b)
    f95[0]   = mult_exp(xkp, f95_a, f95_b)
    f50[0]   = mult_exp(xkp, f50_a, f50_b)
    fsig[0]  = mult_exp(xkp, fsg_a, fsg_b)
    favg[0]  = mult_exp(xkpsc, favg_a, favg_b)

cdef void flux_mtd_3(
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
):
    """
    Method 3: Near linear except exponential f95
    """
    fmean[0] = lin(xkp, fmean_a, fmean_b)
    f95[0]   = lin_exp(xkp, f95_a, f95_b)
    f50[0]   = lin(xkp, f50_a, f50_b)
    fsig[0]  = lin(xkp, fsg_a, fsg_b)
    favg[0]  = lin(xkpsc, favg_a, favg_b)

cdef void flux_mtd_4(
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
):
    """
    Method 4: Near multiplicative exponential except linear f50
    """
    fmean[0] = mult_exp(xkp, fmean_a, fmean_b)
    f95[0]   = mult_exp(xkp, f95_a, f95_b)
    f50[0]   = lin(xkp, f50_a, f50_b)
    fsig[0]  = mult_exp(xkp, fsg_a, fsg_b)
    favg[0]  = mult_exp(xkpsc, favg_a, favg_b)

cdef void flux_mtd_5(
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
):
    """
    Method 5: Near exponential except linear f50, fsig
    """
    fmean[0] = lin_exp(xkp, fmean_a, fmean_b)
    f95[0]   = lin_exp(xkp, f95_a, f95_b)
    f50[0]   = lin(xkp, f50_a, f50_b)
    fsig[0]  = lin(xkp, fsg_a, fsg_b)
    favg[0]  = lin_exp(xkpsc, favg_a, favg_b)

cdef void flux_mtd_6(
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
):
    """
    Method 6: Near linear except exponential f50
    """
    fmean[0] = lin(xkp, fmean_a, fmean_b)
    f95[0]   = lin(xkp, f95_a, f95_b)
    f50[0]   = lin_exp(xkp, f50_a, f50_b)
    fsig[0]  = lin(xkp, fsg_a, fsg_b)
    favg[0]  = lin(xkpsc, favg_a, favg_b)

#: Sector specifications
cdef SectorSpec[3] SECTORS_SOLAR_WIND

SECTORS_SOLAR_WIND[0] = SectorSpec(
    x_left=5.0, x_right=20.0,
    y_bottom=13.0, y_top=30.0,
    fmean_a=0.2477179, f_mean_b=2.532024,
    f95_a=0.1987326, f95_b=3.283149,
    f50_a=1.31762, f50_b=0.127923,
    fsg_a=0.1528122, fsg_b=3.316698,
    favg_a=0.2477179, favg_b=2.532024,
    method=flux_mtd_6
)

SECTORS_SOLAR_WIND[1] = SectorSpec(
    x_left=8.0, x_right=22.0,
    y_bottom=-10.0, y_top=10.0,
    fmean_a=1.053755, fmean_b=0.1816665,
    f95_a=1.23647, f95_b=0.1521542,
    f50_a=0.137769, f50_b=1.508285,
    fsg_a=1.232657, fsg_b=0.127408,
    favg_a=1.053755, favg_b=0.1816665,
    method=flux_mtd_4
)

SECTORS_SOLAR_WIND[2] = SectorSpec(
    x_left=5.0, x_right=20.0,
    y_bottom=-25.0, y_top=-13.0,
    fmean_a=2.418556, fmean_b=0.09242323,
    f95_a=3.129407, f95_b=0.07088557,
    f50_a=0.4192601, f50_b=1.12118,
    fsg_a=0.2818979, fsg_b=2.802846,
    favg_a=2.418556, favg_b=0.09242323,
    method=flux_mtd_5
)

cdef SectorSpec[4] SECTORS_MAGNETOSHEATH

SECTORS_MAGNETOSHEATH[0] = SectorSpec(
    x_left=-20.0, x_right=0.0,
    y_bottom=15.0, y_top=30.0,
    fmean_a=1.006261, fmean_b=0.2358533,
    f95_a=1.242684, f95_b=0.167599,
    f50_a=0.3595895, f50_b=1.485105,
    fsg_a=1.154729, fsg_b=0.1813815,
    favg_a=1.006261, favg_b=0.2358533,
    method=flux_mtd_4
)

SECTORS_MAGNETOSHEATH[1] = SectorSpec(
    x_left=-5.0, x_right=15.0,
    y_bottom=10.0, y_top=23.0,
    fmean_a=1.075755, fmean_b=0.228047,
    f95_a=1.28287, f95_b=0.1789508,
    f50_a=0.4152996, f50_b=1.144803,
    fsg_a=1.231504, fsg_b=0.1678395,
    favg_a=1.075755, favg_b=0.228047,
    method=flux_mtd_4
)

SECTORS_MAGNETOSHEATH[2] = SectorSpec(
    x_left=-5.0, x_right=15.0,
    y_bottom=-10.0, y_top=10.0,
    fmean_a=1.059922, fmean_b=0.2450142,
    f95_a=1.262771, f95_b=0.2122456,
    f50_a=0.2699979, f50_b=1.92621,
    fsg_a=1.170143, fsg_b=0.2329764,
    favg_a=1.059922, favg_b=0.2450142,
    method=flux_mtd_4
)

SECTORS_MAGNETOSHEATH[3] = SectorSpec(
    x_left=-25.0, x_right=5.0,
    y_bottom=-30.0, y_top=-12.0,
    fmean_a=0.8034997, fmean_b=0.2943536,
    f95_a=1.029681, f95_b=0.2502851,
    f50_a=0.6214508, f50_b=0.2912395,
    fsg_a=0.890587, fsg_b=0.2943353,
    favg_a=0.8034997, favg_b=0.2943536,
    method=flux_mtd_2
)

cdef SectorSpec[10] SECTORS_MAGNETOSPHERE

SECTORS_MAGNETOSPHERE[0] = SectorSpec(
    x_left=-10.0, x_right=-6.0,
    y_bottom=-6.0, y_top=4.0,
    fmean_a=2.443395e-1, fmean_b=3.845401,
    f95_a=4.539177, fmean_b=3.807753e-2,
    f50_a=3.058675e-1, fmean_b=3.264229,
    fsg_a=2.038728e-1, fsg_b=4.088105,
    favg_a=2.443395e-1, favg_b=3.845401,
    method=flux_mtd_3
)

SECTORS_MAGNETOSPHERE[1] = SectorSpec(
    x_left=-7.0, x_right=0.0,
    y_bottom=8.0, y_top=13.0,
    fmean_a=1.430943, fmean_b=1.431528e-1,
    f95_a=1.510317, f95_b=1.432958e-1,
    f50_a=1.285142, f50_b=1.935917e-1,
    fsg_a=1.544309, fsg_b=8.922371e-2,
    favg_a=1.430943, favg_b=1.431528e-1,
    method=flux_mtd_2
)

SECTORS_MAGNETOSPHERE[2] = SectorSpec(
    x_left=-18.0, x_right=-12.0,
    y_bottom=-18.0, y_top=-11.0,
    fmean_a=4.0986685e-1, fmean_b=2.083598,
    f95_a=3.677568e-1, f95_b=2.792231,
    f50_a=4.706894e-1, f50_b=1.382612,
    fsg_a=3.606812e-1, fsg_b=2.440727,
    favg_a=4.0986685e-1, favg_b=2.083598,
    method=flux_mtd_1
)

SECTORS_MAGNETOSPHERE[3] = SectorSpec(
    x_left=-28.0, x_right=-21.0,
    y_bottom=-18.0, y_top=-10.0,
    fmean_a=3.249438e-1, fmean_b=2.425203,
    f95_a=2.860754e-1, f95_b=3.135966,
    f50_a=3.848902e-1, f50_b=1.684094,
    fsg_a=2.574007e-1, fsg_b=2.909206,
    favg_a=3.249438e-1, favg_b=2.425203,
    method=flux_mtd_1
)

SECTORS_MAGNETOSPHERE[4] = SectorSpec(
    x_left=-29.0, x_right=-25.0,
    y_bottom=-10.0, y_top=-3.0,
    fmean_a=4.343142e-1, fmean_b=2.180041,
    f95_a=4.123490e-1, f95_b=2.759741,
    f50_a=4.826937e-1, f50_b=1.563440,
    fsg_a=3.533449e-1, fsg_b=2.641683,
    favg_a=4.343142e-1, favg_b=2.180041,
    method=flux_mtd_1
)

SECTORS_MAGNETOSPHERE[5] = SectorSpec(
    x_left=-20.0, x_right=-15.0,
    y_bottom=1.0, y_top=10.0,
    fmean_a=1.270656, fmean_b=2.230935e-1,
    f95_a=1.436059, f95_b=1.577674e-1,
    f50_a=1.051924, f50_b=3.327092e-1,
    fsg_a=1.332255, fsg_b=1.706059e-1,
    favg_a=1.270656, favg_b=2.230935e-1,
    method=flux_mtd_2
)

SECTORS_MAGNETOSPHERE[6] = SectorSpec(
    x_left=-19.0, x_right=-14.0,
    y_bottom=10.0, y_top=19.0,
    fmean_a=3.213946e-1, fmean_b=2.964045,
    f95_a=2.901571e-1, f95_b=3.628429,
    f50_a=3.349902e-1, f50_b=2.415445,
    fsg_a=2.934793e-1, fsg_b=3.251733,
    favg_a=3.213946e-1, favg_b=2.964045,
    method=flux_mtd_1
)

SECTORS_MAGNETOSPHERE[7] = SectorSpec(
    x_left=-30.0, x_right=-24.0,
    y_bottom=10.0, y_top=17.0,
    fmean_a=1.114799, fmean_b=2.013791e-1,
    f95_a=1.283152, f95_b=1.715240e-1,
    f50_a=8.948732e-1, f50_b=2.100576e-1,
    fsg_a=1.232212, fsg_b=1.728799e-1,
    favg_a=1.114799, favg_b=2.013791e-1,
    method=flux_mtd_2
)

SECTORS_MAGNETOSPHERE[8] = SectorSpec(
    x_left=0.0, x_right=8.0,
    y_bottom=8.0, y_top=13.0,
    fmean_a=2.373299e-1, fmean_b=3.988981,
    f95_a=1.769049e-1, f95_b=4.660334,
    f50_a=2.427994e-1, f50_b=3.717080,
    fsg_a=2.079478e-1, fsg_b=4.169556,
    favg_a=2.373299e-1, favg_b=3.988981,
    method=flux_mtd_1
)

SECTORS_MAGNETOSPHERE[9] = SectorSpec(
    x_left=-8.0, x_right=-6.0,
    y_bottom=-16.0, y_top=-14.0,
    fmean_a=2.373299e-1, fmean_b=3.988981,
    f95_a=1.769049e-1, f95_b=4.660334,
    f50_a=2.427994e-1, f50_b=3.717080,
    fsg_a=2.079478e-1, fsg_b=4.169556,
    favg_a=2.373299e-1, favg_b=3.988981,
    method=flux_mtd_1
)

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
):
    """
    compute the proton flux vs kp scaling for the section

    inputs: xkp     --- user selected kp index (real value between 0 & 9).
            xkpsc   --- the kp value at the midpoint of the data interval.
            xpar    --- parameter set of sector limit in x
            ypar    --- parameter set of sector limit in y
            fmnp    --- parameter set of fmean computation
            f95p    --- parameter set of f95 computation
            f50p    --- parameter set of f50 computation
            fsgp    --- parameter set of fsig computation
            favp    --- parameter set of favg computation

    output: scmean  --- mean flux scale factor.
            sc95    --- 95% flux scale factor.
            sc50    --- 50% flux scale factor.
            scsig   --- standard deviation scale factor of flux.
            xcen    --- sector center's x-coordinate (re).
            ycen    --- sector center's x-coordinate (re).
    """

    cdef double fmean, f95, f50, fsig, favg

    spec.method(
        xkp, xkpsc,
        spec.x_left, spec.x_right,
        spec.f95_a, spec.f95_b,
        spec.f50_a, spec.f50_b,
        spec.fsg_a, spec.fsg_b,
        spec.favg_a, spec.favg_b,
        &fmean, &f95, &f50, &fsig, &favg
    )

    scmean[0] = 10**(fmean / favg)
    sc95[0]   = 10**(f95   / favg)
    sc50[0]   = 10**(f50   / favg)
    scsig[0]  = 10**(fsig  / favg)

    xcen[0] = cent(spec.x_left + spec.x_right)
    ycen[0] = cent(spec.y_bottom + spec.y_top)
