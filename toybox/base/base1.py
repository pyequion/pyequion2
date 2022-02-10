# -*- coding: utf-8 -*-

#import sys
#sys.path.insert(0,'../..')

import numpy as np

from pyequion import EquilibriumSystem


eqsys = EquilibriumSystem(['C','Ca'], from_elements=True, activity_model="DEBYE")
elements_balance = {'Ca':0.028, 'C':0.065}
#species_balance = {'Ca++':0.028, 'Cl-':0.056, 'Na+':0.075, 'HCO3-':0.065}
TK = 273.15 + 95.0
solution, res = eqsys.solve_equilibrium_elements_balance(TK, elements_balance, tol=1e-12)
solution.savelog("solutionlog.txt")