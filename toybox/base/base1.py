# -*- coding: utf-8 -*-

#import sys
#sys.path.insert(0,'../..')

import numpy as np

from pyequion import EquilibriumSystem
from pyequion import builder
from pyequion import converters


eqsys = EquilibriumSystem(['C','Ca'], from_elements=True, activity_model="PITZER")
elements_balance = {'Ca':0.028, 'C':0.065}
#species_balance = {'Ca++':0.028, 'Cl-':0.056, 'Na+':0.075, 'HCO3-':0.065}
TK = 273.15 + 95.0
solution1, res = eqsys.solve_equilibrium_elements_balance(TK, elements_balance, tol=1e-12)
#print(solution1)
#print(solution.saturation_indexes)
#raise KeyError
#solution2, res = eqsys.solve_equilibrium_elements_balance_phases(TK, elements_balance, tol=1e-12, has_gas_phases=True,
#                                                                 PATM=10.0)
#solution2, res = eqsys.solve_equilibrium_elements_balance_phases(TK, elements_balance, tol=1e-12, solid_phases=['Halite', 'Calcite'],
#                                                                 has_gas_phases=True)
solution3, res = eqsys.solve_equilibrium_elements_balance_phases(TK, elements_balance, solid_phases=[], tol=1e-12, PATM=1e-2)