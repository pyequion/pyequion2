# -*- coding: utf-8 -*-

from pyequion2 import InterfaceSystem


intsys = InterfaceSystem(['Ca', 'C', 'Na', 'Cl', 'Mg'], from_elements=True)

elements_balance = {'Ca':0.028, 'C':0.065, 'Na':0.075, 'Cl':0.056, 'Mg':0.02}
TK = 298.15
solution, res = intsys.solve_equilibrium_mixed_balance(TK, molal_balance=elements_balance)

intsys.set_interface_phases(phases=['Calcite'], fill_defaults=True)
molals_bulk = solution.solute_molals

transport_params = {'type': 'pipe',
                    'shear_velocity': 1e-6}

solution_int, res = intsys.solve_interface_equilibrium(TK,
                                                       molals_bulk,
                                                       transport_params,
                                                       fully_diffusive=False)