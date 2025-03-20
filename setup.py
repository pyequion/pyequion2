# -*- coding: utf-8 -*-
import warnings
from setuptools import setup, find_packages

packages = ['pyequion2'] + \
           ['pyequion2.' + subpack for subpack in find_packages('pyequion2')]

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

# Optional dependencies
extras_require = {
    "gui": ["PyQt5>=5.15.6"]
}

# Setup common arguments
common_args = dict(
    name="pyequion2",
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
    install_requires=REQUIREMENTS,
    extras_require=extras_require,
)

try:
    import numpy
    import Cython
    from Cython.Build import cythonize
    
    # Define the extension module
    from setuptools.extension import Extension
    ext = Extension(
        "pyequion2.cythonize_module",
        ["pyequion2/cythonize_module.pyx"],
    )
    
    setup(
        version="0.0.6.4",
        setup_requires=['setuptools>=18.0', 'cython'],
        ext_modules=cythonize([ext]),
        include_dirs=[numpy.get_include()],
        **common_args
    )
except Exception as e:
    warnings.warn(f"Could not install with cython module. Installing pure python. Error: {e}")
    setup(
        version="0.0.6.3",
        **common_args
    )
