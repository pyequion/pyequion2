# -*- coding: utf-8 -*-

#import sys
#sys.path.insert(0,'../..')

import numpy as np

from pyequion import EquilibriumSystem


eqsys = EquilibriumSystem(['C','Ca'], from_elements=True, activity_model="PITZER")
molal_balance = {'Ca':0.028, 'C':0.065}
TK = 298.15
PATM = 1.0
solution, _ = eqsys.solve_equilibrium_mixed_balance(TK, molal_balance=molal_balance, PATM=PATM)
TK_lb = 273.15
TK_ub = TK_lb + 100
npoints = 10
solution_list = eqsys.solve_equilibrium_elements_balance_phases_sequential((TK_lb, TK_ub),
                                                                 molal_balance,
                                                                 PATM=PATM,
                                                                 npoints=10)