# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../..')

import numpy as np

from pyequion2 import EquilibriumSystem
from pyequion2 import builder
from pyequion2 import converters

eqsys = EquilibriumSystem(['C','Ca','Na','Cl'], from_elements=True)
c = np.ones(eqsys.nsolutes)*10
x = converters.mmolar_to_molal(c)
y = eqsys.activity_function(x,298.15)

elements_balance = {'Ca':0.028, 'Cl':0.056, 'Na':0.075, 'C':0.065}
species_balance = {'Ca++':0.028, 'Cl-':0.056, 'Na+':0.075, 'HCO3-':0.065}
TK = 298.15
solution,res = eqsys.solve_equilibrium_elements_balance_solids(TK, elements_balance, tol=1e-12)