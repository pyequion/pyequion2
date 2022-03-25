# -*- coding: utf-8 -*-
import warnings
from setuptools import setup, find_packages, Extension, dist
import pathlib

import numpy
import numpy as np

from Cython.Build import cythonize


here = pathlib.Path(__file__).parent.resolve()

cyloc1 = str(here/'pyequion2/activity/coo_tensor_ops/coo_tensor_ops.pyx')

ext = Extension('pyequion2.activity.coo_tensor_ops.coo_tensor_ops',
                sources=[cyloc1])
packages = ['pyequion2'] + \
           ['pyequion2.' + subpack for subpack in find_packages('pyequion2')]

try:
    setup(
        name="pyequion2",
        version="0.0.6.1",
        description="Chemical equilibrium for electrolytes in pure python",
        packages=packages,
        author="PyEquion",
        setup_requires=['setuptools>=18.0', 'cython'],
        ext_modules=cythonize([ext]),
        include_dirs=[numpy.get_include()],
    )
except:
    warnings.warn("Could not install with cython module. Installing pure python")
    setup(
        name="pyequion2",
        version="0.0.6.1",
        description="Chemical equilibrium for electrolytes in pure python",
        packages=packages,
        author="PyEquion",
    )