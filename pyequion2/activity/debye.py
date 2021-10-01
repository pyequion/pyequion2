# -*- coding: utf-8 -*
import functools

import numpy as np

from .. import utils
from .. import constants


def setup_debye(solutes, calculate_osmotic_coefficient=False):
    zarray = np.array([utils.charge_number(specie) for specie in solutes],
                      dtype=np.double)
    f = functools.partial(loggamma_and_osmotic, zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient)

    def g(xarray, TK): return constants.LOG10E * \
        f(xarray, TK)  # ln(gamma) to log10(gamma)
    return g


def loggamma_and_osmotic(carray, T, zarray, calculate_osmotic_coefficient):

    I = 0.5*np.sum(carray*zarray**2)
    sqrtI = np.sqrt(I)
    A = A_debye(T)
    F = A*f_debye(sqrtI)
    logg = zarray**2*F
    resw = -A*sqrtI**3/(1 + constants.B_DEBYE*sqrtI)
    if calculate_osmotic_coefficient:
        osmotic_coefficient = (2*resw/np.sum(carray) + 1)
    else:
        osmotic_coefficient = 1.0
    logg = np.insert(logg, 0, osmotic_coefficient)
    return logg


def A_debye(T):
    Na = 6.0232e23
    ee = 4.8029e-10
    k = 1.38045e-16
    ds = -0.0004 * T + 1.1188
    eer = 305.7 * np.exp(-np.exp(-12.741 + 0.01875 * T) - T / 219.0)
    Aphi = 1.0/3.0*(2.0 * np.pi * Na * ds / 1000) ** 0.5 * \
        (ee / (eer * k * T) ** 0.5) ** 3.0
    return Aphi


def f_debye(sqrtI):
    res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
            2/constants.B_DEBYE*np.log(1 + constants.B_DEBYE*sqrtI))
    return res
