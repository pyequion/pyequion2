import sys
sys.path.insert(0, '../..')

import numpy as np

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
intsys.set_reaction_functions(['linear_ksp'], [[7.64119601e-05]])
molals_bulk = solution.molals
print(solution.concentrations)
transport_dict = {'Ca++': 8.568115256825973e-11,
                  'HCO3-': 1.1166609737092512e-10,
                  'Na+': 1.210000000000001e-10,
                  'Cl-': 1.6037621106854658e-10,
                  'OH-': 3.028357040800728e-10,
                  'H+': 4.425858767009217e-10,
                  'CO2': 9.664503449097217e-11,
                  'CaOH+': 9.664503449097217e-11,
                  'NaOH': 9.664503449097217e-11,
                  'CaHCO3+': 6.374975522141058e-11,
                  'CO3--': 9.698380747882186e-11,
                  'CaCO3': 9.664503449097217e-11,
                  'NaCO3-': 6.994723915431537e-11,
                  'Na2CO3': 9.664503449097217e-11,
                  'NaHCO3': 9.664503449097217e-11}
transport_dict = {k: v*1e3 for k, v in transport_dict.items()}
transport_params = {'type': 'dict',
                    'dict': transport_dict}
#transport_params = {'type': 'pipe',
#                    'shear_velocity': 0.05}
solution_int, res_int = intsys.solve_interface_equilibrium_dr(TK,
                                                              molals_bulk,
                                                              transport_params)
print(solution_int.concentrations)