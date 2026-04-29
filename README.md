# CRMFLX
Python package for computing the Chandra Radiation Model.


## Cython Design Choices

### Python API and C ABI internals:

The original fortran code located all necessary operations in a single fortran source code file, thus the compiled library could only be used for the full algorithmic intention of intaking flux numerical kernel model text files (.asc), the ephemeris data, and then calculating the flux statistics.

At this point, this Cython library rewrite only needs to achieve a similar result, meaning calculating the flux statistics. However, for both clarity and ease of future implementations, this has been compartmentalized into multiple modules with defined python wrapper functions for the important algorithmic steps. These python wrappers are thus accessible when importing the library as a whole, even if they are not used directly by the cronjob script calling the library functions for calcualting the flux statistics.

This means that portions of this library use Cython to define pure C functions (eg bowshk2_c) for internal C kernel usage and the ABI, and then use the Cython `cpdef` function keyword to make use of Cython's reliable python wrapper source generation to define a python wrapper for the API (eg. bowshk2). `cpdef` means that Cython will create a python wrapper and a C function.

We then also include Cython `.pxd` files (which are similar to C header files) for Cython declarations of the `cpdef` defined functions. This is what allows other Cython modules (`.pyx`) to make use of the internal C functionality (with `cminport`) and circumvent expensive Python entry points.

While this defines a sum total of three functions (Python bowshk2, C bowshk2, and C bowshk2_c), this pattern exists to more tightly control data types allowed by the Cython syntax. This also defines a boundary of public API Python functions callable by the interpreter, and private ABI C functions for reuse across the library as a whole. Future implementations and developments should carefully make the distinction of which functions actually need the python overhead and reduce unnecessary exposure of the C kernel internals wherever possible.

### Ctuples and Type Memoryview:
- https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html
- https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
- https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial

Some of the internal C functions make use of the Cython `ctuple` abstraction for return types of certain `cdef` functions (eg. solwind) because these allow for a convenient C source code generation of C struct which mimics the Python tuple. This convention is used for intermediary C fuctions which record intermediary variables used in our calculations. Other internal C functions make use of C pointer outputs (e.g. locreg) and typed memoryview in order to efficiently write to numpy arrays.

Use ctuple for small, internal helper functions. Use pointer outputs for multi-output numerical kernels, especially if they interact with arrays or C code.