# -*- coding: utf-8 -*-
import functools

import numpy as np

from .. import datamods
from .. import utils
from .. import constants


DB_SPECIES = datamods.species['debye']


def setup_extended_debye(solutes, calculate_osmotic_coefficient=False):
    db_species = DB_SPECIES
    I_factor = []
    dh_a = []
    dh_b = []
    for sp in solutes:
        I_factor_, dh_a_, dh_b_ = _species_definition_dh_model(sp, db_species)
        I_factor.append(I_factor_)
        dh_a.append(dh_a_)
        dh_b.append(dh_b_)
    dh_a = np.array(dh_a)
    dh_b = np.array(dh_b)
    I_factor = np.array(I_factor)
    zarray = np.array([utils.charge_number(specie) for specie in solutes])
    g = functools.partial(_loggamma_and_osmotic,
                          zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient,
                          dh_a=dh_a, dh_b=dh_b,
                          I_factor=I_factor)
    return g


def _loggamma_and_osmotic(xarray, TK, zarray, calculate_osmotic_coefficient,
                          dh_a, dh_b, I_factor):
    A, B = _debye_huckel_constant(TK)
    I = 0.5*np.sum(zarray**2*xarray)
    logg1 = -A * zarray ** 2 * \
        np.sqrt(I) / (1 + B * dh_a * np.sqrt(I)) + dh_b * I
    logg2 = I_factor*I
    logg3 = -A * zarray ** 2 * (np.sqrt(I) / (1.0 + np.sqrt(I)) - 0.3 * I)
    logg = np.nan_to_num(logg1)*(~np.isnan(dh_a)) + \
        np.nan_to_num(logg2)*(np.isnan(dh_a) & (~np.isnan(I_factor))) + \
        np.nan_to_num(logg3)*(np.isnan(dh_a) & (np.isnan(I_factor)))
    resw = -A*I**(3/2)/(1 + constants.B_DEBYE*I**(1/2))
    if calculate_osmotic_coefficient:
        osmotic_coefficient = constants.LOG10E*(2*resw/np.sum(xarray)+1)
    else:
        osmotic_coefficient = constants.LOG10E
    # Insertion of osmotic coefficient
    logg = np.insert(logg, 0, osmotic_coefficient)
    return logg


def _debye_huckel_constant(TK):
    epsilon = _dieletricconstant_water(TK)
    rho = _density_water(TK)
    A = 1.82483e6 * np.sqrt(rho) / (epsilon * TK) ** 1.5  # (L/mol)^1/2
    B = 50.2916 * np.sqrt(rho / (epsilon * TK))  # Angstrom^-1 . (L/mol)^1/2
    return A, B


def _species_definition_dh_model(tag, species_activity_db):
    z = utils.charge_number(tag)
    if tag not in species_activity_db:
        if z == 0:
            I_factor = 0.1
            dh_a = np.nan
            dh_b = np.nan
        else:  # Else should use davies
            I_factor = np.nan
            dh_a = np.nan
            dh_b = np.nan
    else:
        db_specie = species_activity_db[tag]
        try:
            if "I_factor" in db_specie:
                I_factor = db_specie["I_factor"]
                dh_a = np.nan
                dh_b = np.nan
            else:
                I_factor = np.nan
                dh_a = db_specie["dh"]["phreeqc"]["a"]
                dh_b = db_specie["dh"]["phreeqc"]["b"]
        except KeyError as e:
            print("Error getting activity of specie = {}".format(tag))
            raise e
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
