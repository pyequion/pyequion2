import sys
sys.path.insert(0, '../..')

import numpy as np
import scipy.linalg

from pyequion import EquilibriumSystem
from pyequion import builder
from pyequion import converters
from pyequion import InterfaceSystem


intsys = InterfaceSystem(['Ca', 'C', 'Na', 'Cl'], from_elements=True,
                         activity_model="PITZER")
elements_balance = {'Ca':0.028, 'C':0.065, 'Na':0.075, 'Cl':0.056}
#species_balance = {'Ca++':0.028, 'Cl-':0.056, 'Na+':0.075, 'HCO3-':0.065}
TK = 298.15
solution,res = intsys.solve_equilibrium_elements_balance(TK, elements_balance, tol=1e-12)
#print(solution.saturation_indexes)
intsys.set_interface_phases(['Calcite'], TK)
intsys.set_reaction_functions({})
intsys.set_reaction_functions({'Calcite': ('linear', [7.64119601e2], None)})

molals_bulk = solution.molals
transport_params = {'type': 'pipe',
                    'shear_velocity': 0.05}
solution_int, res_int = intsys.solve_interface_equilibrium(TK,
                                                           molals_bulk,
                                                           transport_params)
#print(solution_int.concentrations