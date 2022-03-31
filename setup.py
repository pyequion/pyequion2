# -*- coding: utf-8 -*-
import warnings
from setuptools import setup, find_packages, Extension, dist
import pathlib

dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])

import numpy
import numpy as np

from Cython.Build import cythonize


here = pathlib.Path(__file__).parent.resolve()

cyloc1 = str(here/'pyequion2/activity/coo_tensor_ops/coo_tensor_ops.pyx')

ext = Extension('pyequion2.activity.coo_tensor_ops.coo_tensor_ops',
                sources=[cyloc1])
packages = ['pyequion2'] + \
           ['pyequion2.' + subpack for subpack in find_packages('pyequion2')]

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

try:
    setup(
        name="pyequion2",
        version="0.0.6.3",
        description="Chemical equilibrium for electrolytes in pure python",
        packages=packages,
        author="PyEquion",
        setup_requires=['setuptools>=18.0', 'cython'],
        ext_modules=cythonize([ext]),
        include_dirs=[numpy.get_include()],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
        url="https://github.com/pyequion/pyequion2/",
        python_requires=">=3.6",
        install_requires=REQUIREMENTS
    )
except:
    warnings.warn("Could not install with cython module. Installing pure python")
    setup(
        name="pyequion2",
        version="0.0.6.3",
        description="Chemical equilibrium for electrolytes in pure python",
        packages=packages,
        author="PyEquion",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
        url="https://github.com/pyequion/pyequion2/",
        python_requires=">=3.6",
        include_dirs=[numpy.get_include()],
        install_requires=REQUIREMENTS
    )