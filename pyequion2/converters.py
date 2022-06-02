# -*- coding: utf-8 -*-
import periodictable
import numpy as np

from . import water_properties
from . import builder
from .datamods import reactions_solids
from .datamods import density_solids


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


def get_activity_from_fugacity(fugacity, spec, TK=298.15):
    gases =  builder.DEFAULT_DB_FILES["gases"]
    possible = [g for g in gases if spec in g and spec + '(g)' in g]
    if possible == []:
        return None
    else:
        logK = builder.get_log_equilibrium_constants(possible, TK, 1.0)[0]
        logact = logK + np.log10(fugacity)
        return 10**logact
    
def get_activity_from_partial_pressure(pp, spec, TK=298.15):
    #For now, alias for get_activity_from_fugacity
    return get_activity_from_fugacity(pp, spec, TK)
    
def phase_to_molar_weight(phase_name):
    possible_reactions = list(filter(lambda r : r['phase_name'] == phase_name,
                                  reactions_solids))
    if len(possible_reactions) == 0:
        return None
    else:
        reaction = possible_reactions[0]
        specs_and_coefs = [(s, k) for s, k in reaction.items() if k == 1.0]
        specs, coefs = zip(*specs_and_coefs)
        els_and_coefs = builder.get_elements_and_their_coefs(specs)
        spec_masses = [sum([ELEMENTS_MOLAR_WEIGHTS[el]*coef
                          for el, coef in el_and_coefs])
                       for el_and_coefs in els_and_coefs]
        molar_mass = sum([sp_mass*coef for sp_mass, coef in zip(spec_masses, coefs)])
        return molar_mass*1e-3 #g/mol to kg/mol


def phase_density(phase_name):
    return density_solids.densities.get(phase_name, 3000.0)
        
        