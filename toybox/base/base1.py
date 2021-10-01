# -*- coding: utf-8 -*-

#import sys
#sys.path.insert(0,'../..')

import numpy as np

from pyequion import EquilibriumSystem
from pyequion import builder
from pyequion import converters


eqsys = EquilibriumSystem(['C','Ca','Na','Cl'], from_elements=True, activity_model="PITZER")
elements_balance = {'Ca':0.028, 'Cl':0.056, 'Na':0.075, 'C':0.065}
#species_balance = {'Ca++':0.028, 'Cl-':0.056, 'Na+':0.075, 'HCO3-':0.065}
TK = 298.15
solution,res = eqsys.solve_equilibrium_elements_balance(TK, elements_balance, tol=1e-12)
#print(solution.saturation_indexes)