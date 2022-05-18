# -*- coding: utf-8 -*-
import periodictable
from . import water_properties
from . import builder


ELEMENTS_MOLAR_WEIGHTS = {
    el.symbol: el.mass for el in periodictable.elements if el.symbol != 'n'}


def mmolar_to_molal(x, TK=298.15):
    #mol/m^3 = mol/(1e3*L) = 1e-3*mol/L = mmol/L
    return x/water_properties.water_density(TK)


def molal_to_mmolar(x, TK=298.15):
    return x*water_properties.water_density(TK) #mmol/L = mol/m^3 to mol/kg


def mgl_to_molal(x, specie, TK=298.15):
    #x : mg/L
    #mg/L = (1e-3*g)/(1e-3*m^3) = g/m^3
    gkg = x/water_properties.water_density(TK)  # g/(kg H2O)
    if specie in ELEMENTS_MOLAR_WEIGHTS:
        molar_mass = ELEMENTS_MOLAR_WEIGHTS[specie]
    else: #Specie
        el_and_coefs = builder.get_elements_and_their_coefs([specie])[0]
        molar_mass = sum([ELEMENTS_MOLAR_WEIGHTS[el]*coef
                          for el, coef in el_and_coefs])  # g/mol
    molal = gkg/molar_mass #mol/(kg H20)
    return molal


def molal_to_mgl(x, specie, TK=298.15):
    if specie in ELEMENTS_MOLAR_WEIGHTS:
        molar_mass = ELEMENTS_MOLAR_WEIGHTS[specie]
    else: #Specie
        el_and_coefs = builder.get_elements_and_their_coefs([specie])[0]
        molar_mass = sum([ELEMENTS_MOLAR_WEIGHTS[el]*coef
                          for el, coef in el_and_coefs])  # g/mol
    y = x*molar_mass #mol/(kg H2O) to g/(kg H2O)
    mgl = y*water_properties.water_density(TK) #g/(kg H2O) to g/m^3 = mg/L
    return mgl