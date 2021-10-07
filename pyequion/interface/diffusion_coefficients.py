# -*- coding: utf-8 -*-
import numpy as np
from ..water_properties import water_dynamic_viscosity


def diff_temp_formula(TK, dw0, dw1):
    return dw0*np.exp(dw1/TK - dw1/298.15) *\
        TK*water_dynamic_viscosity(298.15)/(298.15*water_dynamic_viscosity(TK))


def diffusion_temp(coefficients, TK):
    coefficients_temp = dict()
    for key, value in coefficients.items():
        if len(value) < 2:
            coefficients_temp[key] = diff_temp_formula(
                TK, coefficients[key][0], 0.0)
        else:
            coefficients_temp[key] = diff_temp_formula(
                TK, coefficients[key][0], coefficients[key][1])
    return coefficients_temp


def diffusion_temp_istrength(coefficients, TK, I):
    raise NotImplementedError


COEFFICIENTS = \
    {'Na+': (1.331e-09, 122.0, 1.52, 3.7),
     'HCO3-': (1.18e-09,),
     'Ca++': (7.931e-10, 97.0, 3.4, 24.6),
     'Cl-': (2.031e-09, 194.0, 1.6, 6.9),
     'OH-': (5.27e-09,),
     'H+': (9.311e-09, 1000.0, 0.46, 1.1e-09),
     'CO3--': (9.551e-10, 0.0, 1.12, 2.84),
     'NaCO3-': (5.85e-10,),
     'CaHCO3+': (5.09e-10,),
     'Mg++': (7.051e-10, 111.0, 2.4, 13.7),
     'K+': (1.961e-09, 395.0, 2.5, 21.0),
     'Fe++': (7.191e-10,),
     'Mn++': (6.881e-10,),
     'Al+++': (5.591e-10,),
     'Ba++': (8.481e-10, 46.0),
     'Sr++': (7.941e-10, 161.0),
     'H4SiO4': (1.101e-09,),
     'SO4--': (1.071e-09, 34.0, 2.08, 13.4),
     'NO3-': (1.91e-09, 184.0, 1.85, 3.85),
     'H3BO3': (1.11e-09,),
     'PO4---': (6.121e-10,),
     'F-': (1.461e-09,),
     'Li+': (1.031e-09, 80.0),
     'Br-': (2.011e-09, 258.0),
     'Zn++': (7.151e-10,),
     'Cd++': (7.171e-10,),
     'Pb++': (9.451e-10,),
     'Cu++': (7.331e-10,)}
