"""
This script is used to build the bowshock module separately from the main CRMFLX package for development purposes.
https://cython.readthedocs.io/en/latest/src/tutorial/external.html#id5

As part of the compilation process, run with the setuptool conventsion rather than the build library conventions.
    >>> python setup_bowshock.py build_ext --inplace
"""
from setuptools import Extension, setup
from pathlib import Path
#: Determination of environment libraries depending on build platform.
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

#: pyproject.toml tool.setuptools.packages.find config specifying "src"
#: as the source to build packages, but this config is separate from
#: cythonize build path config. Must specify twice.
if USE_CYTHON and Path("src/crmflx/bowshock.pyx").exists():
    ext = ".pyx"
else:
    ext = ".c"

extensions = [
    Extension("crmflx.bowshock",
              sources=[f"src/crmflx/bowshock{ext}"],
              libraries=["m"] #Unit-like spcific
              )
]

if USE_CYTHON and Path("src/crmflx/bowshock.pyx").exists():
    extensions = cythonize(extensions, force=True)

setup(
    name="crmflx-bowshock-dev",
    ext_modules=extensions
)