from setuptools import Extension, setup
from Cython.Build import cythonize

#: pyproject.toml tool.setuptools.packages.find config specifying "src"
#: as the source to build packages, but this config is separate from
#: cythonize build path config. Must specify twice.

extensions = [
    Extension("crmflx.core", ["src/crmflx/core.pyx"])
]

setup(
    ext_modules = cythonize(extensions, force=True)
)