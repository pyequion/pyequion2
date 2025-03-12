# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

packages = ['pyequion2'] + \
           ['pyequion2.' + subpack for subpack in find_packages('pyequion2')]

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

try:
    setup(
        name="pyequion2",
        version="0.0.6.4",
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
        install_requires=REQUIREMENTS,
        extras_require={
            "gui": ["PyQt5>=5.15.6"]
        },
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