# -*- coding: utf-8 -*-
import periodictable
from . import water_properties
from . import builder


ELEMENTS_MOLAR_WEIGHTS = {
    el.symbol: el.mass for el in periodictable.elements if el.symbol != 'n'}


def mmolar_to_molal(x, TK=298.15):
    return x/water_properties.water_density(TK)


def molal_to_mmolar(x, TK=298.15):
    return x*water_properties.water_density(TK)


def mgl_to_molal(x, specie, TK=298.15):
    mgkg = x*water_properties.water_density(TK)  # mg/kg
    gkg = 1e-3*mgkg  # g/kg
    el_and_coefs = builder.get_elements_and_their_coefs([specie])[0]
    molar_mass = sum([ELEMENTS_MOLAR_WEIGHTS[el]*coef
                      for el, coef in el_and_coefs])  # g/mol
    molal = gkg/molar_mass
    return molal
