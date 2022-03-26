# -*- coding: utf-8 -*-
"""
A pure python implementation for electrolytes chemical equilibrium.

Repository can be found at
https://github.com/pyequion/pyequion2

Example
-------
>>> import pyequion
>>> #Define equilibrium calculation class
>>> eqsys = pyequion.EquilibriumSystem(['HCO3-','Ca++','Na+','Cl-'])
>>> #Define mass balances and system temperature
>>> elements_balance = {'Ca':0.028, 'Cl':0.056, 'Na':0.075, 'C':0.065}
>>> TK = 298.15
>>> #Solve equilibrium
>>> solution, solution_stats = eqsys.solve_equilibrium_elements_balance(TK, elements_balance, tol=1e-12)
>>> #Show properties
>>> solution.ph
7.5660993446870854

GUI Example
-----------
>>> import pyequion
>>> pyequion.rungui()
"""

from .equilibrium_system import EquilibriumSystem
from .interface import InterfaceSystem
from .gui import run as rungui
from . import converters
from . import water_properties

#__all__ = ['EquilibriumSystem', 'InterfaceSystem', 'converters', 'water_properties']