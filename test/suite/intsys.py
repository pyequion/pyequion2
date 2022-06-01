# -*- coding: utf-8 -*-

from pyequion2 import InterfaceSystem

def test1():
    intsys = InterfaceSystem(['Ca', 'C', 'Na', 'Cl', 'Mg'], from_elements=True)
    
    elements_balance = {'Ca':0.028, 'C':0.065, 'Na':0.075, 'Cl':0.056, 'Mg':0.02}
    TK = 298.15
    solution, res = intsys.solve_equilibrium_mixed_balance(TK, molal_balance=elements_balance)
    
    #Run 1a
    intsys.set_interface_phases(phases=['Calcite'], fill_defaults=True)
    molals_bulk = solution.solute_molals
    transport_params = {'type': 'pipe',
                        'shear_velocity': 0.01}
    solution_int, res = intsys.solve_interface_equilibrium(TK,
                                                           molals_bulk,
                                                           transport_params,
                                                           fully_diffusive=False,
                                                           transport_model=None)
    
    #Run 1b
    intsys.set_global_transport_model("B")
    solution_int, res = intsys.solve_interface_equilibrium(TK,
                                                           molals_bulk,
                                                           transport_params,
                                                           fully_diffusive=False,
                                                           transport_model="A")
    
    #Run 1c
    intsys.set_interface_phases()
    transport_params = {'type': 'sphere',
                        'radius': 1e-6}
    solution_int, res = intsys.solve_interface_equilibrium(TK,
                                                           molals_bulk,
                                                           transport_params,
                                                           fully_diffusive=True,
                                                           transport_model=None)

test1()