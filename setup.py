#!/usr/bin/env /data/mta/Script/Python3.6/envs/ska3/bin/python
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='CRM FLX',
    ext_modules=cythonize("crmflx.pyx"),
    zip_safe=False,
)
