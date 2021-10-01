# -*- coding: utf-8 -*-
import os
import pathlib

from setuptools import setup
from Cython.Build import cythonize

loc = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__)))/"coo_tensor_ops.pyx")
setup(
    ext_modules = cythonize(loc)
)