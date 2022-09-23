# -*- coding: utf-8 -*-
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

from .. import datamods
from .. import utils
from .. import constants


DB_SPECIES = datamods.species['debye']


def setup_extended_debye(solutes,
                         calculate_osmotic_coefficient=False,
                         backend='numpy'):
    assert backend in ['numpy', 'jax', "torch"]
    if backend == 'jax':
        np_ = jnp #Hack
    else:
        np_ = np
    db_species = DB_SPECIES
    I_factor = []
    dh_a = []
    dh_b = []
    for sp in solutes:
        I_factor_, dh_a_, dh_b_ = _species_definition_dh_model(sp, db_species, backend=backend)
        I_factor.append(I_factor_)
        dh_a.append(dh_a_)
        dh_b.append(dh_b_)
    if backend in ["numpy", "jax"]:
        dh_a = np_.array(dh_a)
        dh_b = np_.array(dh_b)
        I_factor = np_.array(I_factor)
        zarray = np_.array([utils.charge_number(specie) for specie in solutes])
    else:
        dh_a = torch.tensor(dh_a, dtype=torch.float)
        dh_b = torch.tensor(dh_b, dtype=torch.float)
        I_factor = torch.tensor(I_factor, dtype=torch.float)
        zarray = torch.tensor([utils.charge_number(specie) for specie in solutes], dtype=torch.float)
    g = functools.partial(_loggamma_and_osmotic,
                          zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient,
                          dh_a=dh_a, dh_b=dh_b,
                          I_factor=I_factor,
                          backend=backend) #loggamma
    if backend == 'jax':
        g = jax.jit(g)
    return g


def _loggamma_and_osmotic(xarray, TK, zarray, calculate_osmotic_coefficient,
                          dh_a, dh_b, I_factor,
                          backend = "numpy"):
    if backend in ["numpy", "torch"]:
        np_ = np
    else:
        np_ = jnp
    if backend in ["numpy", "jax"]:
        A, B = _debye_huckel_constant(TK, backend=backend)
        I = 0.5*np_.sum(zarray**2*xarray, axis=-1, keepdims=True)
        logg1 = -A * zarray ** 2 * \
            np_.sqrt(I) / (1 + B * dh_a * np_.sqrt(I)) + dh_b * I
        logg2 = I_factor*I
        logg3 = -A * zarray ** 2 * (np_.sqrt(I) / (1.0 + np_.sqrt(I)) - 0.3 * I)
        logg = np_.nan_to_num(logg1)*(~np_.isnan(dh_a)) + \
            np_.nan_to_num(logg2)*(np_.isnan(dh_a) & (~np_.isnan(I_factor))) + \
            np_.nan_to_num(logg3)*(np_.isnan(dh_a) & (np_.isnan(I_factor)))
        resw = -A*I**(3/2)/(1 + constants.B_DEBYE*I**(1/2))
        if calculate_osmotic_coefficient:
            osmotic_coefficient = constants.LOG10E*(2*resw/np_.sum(xarray, axis=-1, keepdims=True)+1)
        else:
            osmotic_coefficient = constants.LOG10E
        osmotic_coefficient = np_.ones(logg.shape[:-1])*constants.LOG10E
        # Insertion of osmotic coefficient
        logg = np_.hstack([osmotic_coefficient, logg])
    else:
        A, B = _debye_huckel_constant(TK, backend=backend)
        
    return logg

def _debye_huckel_constant(TK, backend="numpy"):
    if backend in ["numpy", "torch"]:
        np_ = np
    else:
        np_ = jnp
    epsilon = _dieletricconstant_water(TK)
    rho = _density_water(TK)
    A = 1.82483e6 * np_.sqrt(rho) / (epsilon * TK) ** 1.5  # (L/mol)^1/2
    B = 50.2916 * np_.sqrt(rho / (epsilon * TK))  # Angstrom^-1 . (L/mol)^1/2
    if backend == "torch":
        A = float(A)
        B = float(B)
    return A, B


def _species_definition_dh_model(tag, species_activity_db, backend="numpy"):
    if backend in ["numpy", "torch"]:
        np_ = np
    else:
        np_ = jnp
    z = utils.charge_number(tag)
    if tag not in species_activity_db:
        if z == 0:
            I_factor = 0.1
            dh_a = np_.nan
            dh_b = np_.nan
        else:  # Else should use davies
            I_factor = np_.nan
            dh_a = np_.nan
            dh_b = np_.nan
    else:
        db_specie = species_activity_db[tag]
        try:
            if "I_factor" in db_specie:
                I_factor = db_specie["I_factor"]
                dh_a = np_.nan
                dh_b = np_.nan
            else:
                I_factor = np_.nan
                dh_a = db_specie["dh"]["phreeqc"]["a"]
                dh_b = db_specie["dh"]["phreeqc"]["b"]
        except KeyError as e:
            print("Error getting activity of specie = {}".format(tag))
            raise e
    if backend == "torch":
        I_factor = float(I_factor)
        dh_a = float(dh_a)
        dh_b = float(dh_b)
    return I_factor, dh_a, dh_b


def _dieletricconstant_water(TK):
    # for TK: 273-372
    return 0.24921e3 - 0.79069 * TK + 0.72997e-3 * TK ** 2


def _density_water(TK):
    # for TK: 273-372
    return (
        0.183652
        + 0.00724987 * TK
        - 0.203449e-4 * TK ** 2
        + 1.73702e-8 * TK ** 3
    )
