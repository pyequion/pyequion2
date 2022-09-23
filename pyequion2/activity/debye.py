# -*- coding: utf-8 -*
import functools
import warnings

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except (ImportError, AssertionError):
    warnings.warn("JAX not installed, so can't be used as backend")

from .. import utils
from .. import constants


def setup_debye(solutes, calculate_osmotic_coefficient=False, backend='numpy'):
    assert backend in ['numpy', 'jax']
    if backend == 'jax':
        #A small hack here
        np_ = jnp
        zarray = np_.array([utils.charge_number(specie) for specie in solutes],
                          dtype=np_.float32)
    else:
        np_ = np
        zarray = np_.array([utils.charge_number(specie) for specie in solutes],
                           dtype=np_.double)
    f = functools.partial(loggamma_and_osmotic, zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient,
                          np_=np_)
    def g(xarray, TK): return constants.LOG10E * \
        f(xarray, TK)  # ln(gamma) to log10(gamma)
    if backend == 'jax':
        g = jax.jit(g)
    return g


def loggamma_and_osmotic(carray, T, zarray, calculate_osmotic_coefficient, np_=np):

    I = 0.5*np_.sum(carray*zarray**2, axis=-1, keepdims=True)
    sqrtI = np_.sqrt(I)
    A = A_debye(T, np_=np_)
    F = A*f_debye(sqrtI, np_=np_)
    logg = zarray**2*F
    resw = -A*sqrtI**3/(1 + constants.B_DEBYE*sqrtI)
    if calculate_osmotic_coefficient:
        osmotic_coefficient = (2*resw/np_.sum(carray) + 1)
    else:
        osmotic_coefficient = 1.0
        if carray.ndim > 1:
            osmotic_coefficient = osmotic_coefficient*np_.ones((carray.shape[0], 1))
    if carray.ndim > 1:
        logg = np_.hstack([osmotic_coefficient, logg])
    else:
        logg = np_.insert(logg, 0, osmotic_coefficient)
    return logg


def A_debye(T, np_=np):
    Na = 6.0232e23
    ee = 4.8029e-10
    k = 1.38045e-16
    ds = -0.0004 * T + 1.1188
    eer = 305.7 * np_.exp(-np_.exp(-12.741 + 0.01875 * T) - T / 219.0)
    Aphi = 1.0/3.0*(2.0 * np_.pi * Na * ds / 1000) ** 0.5 * \
        (ee / (eer * k * T) ** 0.5) ** 3.0
    return Aphi


def f_debye(sqrtI, np_=np):
    res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
            2/constants.B_DEBYE*np_.log(1 + constants.B_DEBYE*sqrtI))
    return res
