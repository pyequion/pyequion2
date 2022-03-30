# -*- coding: utf-8 -*-

from pyequion2 import InterfaceSystem


intsys = InterfaceSystem(['Ca', 'C', 'Na', 'Cl'], from_elements=True)

elements_balance = {'Ca':0.028, 'C':0.065, 'Na':0.075, 'Cl':0.056}
TK = 298.15
solution, res = intsys.solve_equilibrium_elements_balance(TK, elements_balance)
