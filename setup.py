# -*- coding: utf-8 -*-

from setuptools import setup, find_packages, Extension, dist
import pathlib

import numpy

#import numpy
from Cython.Build import cythonize


here = pathlib.Path(__file__).parent.resolve()

#long_description = (here / 'README.md').read_text(encoding='utf-8')
#coo_tensor_ops_loc = (here/'pyequion/activity/coo_tensor_ops/coo_tensor_ops.pyx').read_text(encoding='utf-8')
cyloc1 = str(here/'pyequion/activity/coo_tensor_ops/coo_tensor_ops.pyx')

ext = Extension('pyequion.activity.coo_tensor_ops.coo_tensor_ops',
                sources=[cyloc1])

setup(
    name="pyequion",
    version="0.0.6.1",
    description="Chemical equilibrium for electrolytes in pure python",
#    packages=['pyequion'],
    packages=find_packages(include=['pyequion']),
#    package_dir={"":"pyequion"},
    author="NIDF Scaling",
    author_email="nidf.scaling@gmail.com",
    url="https://github.com/nidf-scaling/pyequion/",
    setup_requires=['setuptools>=18.0', 'cython'],
    ext_modules=cythonize([ext]),
    include_dirs=[numpy.get_include()],
)
