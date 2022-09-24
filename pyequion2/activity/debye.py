# -*- coding: utf-8 -*
import functools
import warnings

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except (ImportError, AssertionError):
    warnings.warn("JAX not installed, so can't be used as backend")
try:
    import torch
except (ImportError, AssertionError):
    warnings.warn("PyTorch not installed, so can't be used as backend")

from .. import utils
from .. import constants


def setup_debye(solutes, calculate_osmotic_coefficient=False, backend='numpy'):
    assert backend in ['numpy', 'jax', "torch"]
    if backend == 'jax':
        zarray = jnp.array([utils.charge_number(specie) for specie in solutes],
                          dtype=jnp.float32)
    elif backend == "numpy":
        zarray = np.array([utils.charge_number(specie) for specie in solutes],
                           dtype=np.double)
    elif backend == "torch":
        zarray = torch.tensor([utils.charge_number(specie) for specie in solutes],
                               dtype=torch.float)        
    f = functools.partial(loggamma_and_osmotic, zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient,
                          backend=backend)
    def g(xarray, TK): return constants.LOG10E * \
        f(xarray, TK)  # ln(gamma) to log10(gamma)
    if backend == 'jax':
        g = jax.jit(g)
    return g


def setup_debye_limiting(solutes, calculate_osmotic_coefficient=False, backend='numpy'):
    assert backend in ['numpy', 'jax', "torch"]
    if backend == 'jax':
        zarray = jnp.array([utils.charge_number(specie) for specie in solutes],
                          dtype=jnp.float32)
    elif backend == "numpy":
        zarray = np.array([utils.charge_number(specie) for specie in solutes],
                           dtype=np.double)
    elif backend == "torch":
        zarray = torch.tensor([utils.charge_number(specie) for specie in solutes],
                               dtype=torch.float)        
    f = functools.partial(loggamma_and_osmotic_limiting, zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient,
                          backend=backend)
    def g(xarray, TK): return constants.LOG10E * \
        f(xarray, TK)  # ln(gamma) to log10(gamma)
    if backend == 'jax':
        g = jax.jit(g)
    return g


def loggamma_and_osmotic(carray, T, zarray, calculate_osmotic_coefficient, backend="numpy"):
    if backend in ["numpy", "torch"]:
        np_ = np
    else:
        np_ = jnp
    if backend == "torch":
        I = 0.5*torch.sum(carray*zarray**2, dim=-1, keepdim=True)  
        sqrtI = torch.sqrt(I)
    else:
        I = 0.5*np_.sum(carray*zarray**2, axis=-1, keepdims=True)
        sqrtI = np_.sqrt(I)
    A = A_debye(T, backend=backend)
    F = A*f_debye(sqrtI, backend=backend)
    logg = zarray**2*F
    resw = -A*sqrtI**3/(1 + constants.B_DEBYE*sqrtI)
    if calculate_osmotic_coefficient:
        if backend == "torch":
            osmotic_coefficient = (2*resw/torch.sum(carray) + 1)
        else:
            osmotic_coefficient = (2*resw/np_.sum(carray) + 1)
    else:
        osmotic_coefficient = 1.0
    if backend == "torch":
        osmotic_coefficient = torch.ones_like(logg[..., :1])*osmotic_coefficient
        logg = torch.cat([osmotic_coefficient, logg], dim=-1)
    else:
        osmotic_coefficient = np_.ones(logg.shape[:-1])*osmotic_coefficient
        logg = np_.hstack([osmotic_coefficient, logg])
    return logg


def loggamma_and_osmotic_limiting(carray, T, zarray, calculate_osmotic_coefficient, backend="numpy"):
    if backend in ["numpy", "torch"]:
        np_ = np
    else:
        np_ = jnp
    if backend == "torch":
        I = 0.5*torch.sum(carray*zarray**2, dim=-1, keepdim=True)  
        sqrtI = torch.sqrt(I)
    else:
        I = 0.5*np_.sum(carray*zarray**2, axis=-1, keepdims=True)
        sqrtI = np_.sqrt(I)
    A = A_debye(T, backend=backend)
    F = A*f_limiting(sqrtI, backend=backend)
    logg = zarray**2*F
    resw = -A*sqrtI**3
    if calculate_osmotic_coefficient:
        if backend == "torch":
            osmotic_coefficient = (2*resw/torch.sum(carray) + 1)
        else:
            osmotic_coefficient = (2*resw/np_.sum(carray) + 1)
    else:
        osmotic_coefficient = 1.0
    if backend == "torch":
        osmotic_coefficient = torch.ones_like(logg[..., :1])*osmotic_coefficient
        logg = torch.cat([osmotic_coefficient, logg], dim=-1)
    else:
        osmotic_coefficient = np_.ones(logg.shape[:-1])*osmotic_coefficient
        logg = np_.hstack([osmotic_coefficient, logg])
    return logg


def A_debye(T, backend="numpy"):
    if backend in ["numpy", "torch"]:
        np_ = np
    else:
        np_ = jnp
    Na = 6.0232e23
    ee = 4.8029e-10
    k = 1.38045e-16
    ds = -0.0004 * T + 1.1188
    eer = 305.7 * np_.exp(-np_.exp(-12.741 + 0.01875 * T) - T / 219.0)
    Aphi = 1.0/3.0*(2.0 * np_.pi * Na * ds / 1000) ** 0.5 * \
        (ee / (eer * k * T) ** 0.5) ** 3.0
    if backend == "torch":
        Aphi = float(Aphi)
    return Aphi


def f_debye(sqrtI, backend="numpy"):
    if backend == "torch":
        res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
                2/constants.B_DEBYE*torch.log(1 + constants.B_DEBYE*sqrtI))
    else:
        if backend == "numpy":
            np_ = np
        elif backend == "jax":
            np_ = jnp
        res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
                2/constants.B_DEBYE*np_.log(1 + constants.B_DEBYE*sqrtI))
    return res


def f_limiting(sqrtI, backend="numpy"):
    if backend == "torch":
        res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
                2/constants.B_DEBYE*torch.log(1 + constants.B_DEBYE*sqrtI))
    else:
        if backend == "numpy":
            np_ = np
        elif backend == "jax":
            np_ = jnp
        res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
                2/constants.B_DEBYE*np_.log(1 + constants.B_DEBYE*sqrtI))
    return res