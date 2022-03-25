# -*- coding: utf-8 -*-
"""
A pure python implementation for electrolytes chemical equilibrium.

Example
-------
>>> import pyequion
>>> #Define equilibrium calculation class
>>> eqsys = pyequion.EquilibriumSystem(['HCO3-','Ca++','Na+','Cl-'])
>>> #Define mass balances and system temperature
>>> elements_balance = {'Ca':0.028, 'Cl':0.056, 'Na':0.075, 'C':0.065}
>>> TK = 298.15
>>> #Solve equilibrium
>>> solution, res = eqsys.solve_equilibrium_elements_balance(TK, elements_balance, tol=1e-12)
>>> #Show residual
array([ 0.00000000e+00,  3.46389584e-14,  6.21724894e-14,  8.47322212e-13,
        1.55431223e-14,  1.33226763e-14, -8.88178420e-15,  5.55111512e-14,
        2.22044605e-14,  1.21125332e-13,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  6.93889390e-18])
>>> #Show properties
>>> solution.molals
{'Ca++': 0.017218806689313353,
 'Na+': 0.07264482713326716,
 'HCO3-': 0.0488901824714975,
 'Cl-': 0.056,
 'OH-': 5.0953557183924075e-06,
 'H+': 3.925292585860418e-09,
 'CO2': 0.00026086514158995384,
 'CaOH+': 4.716449233946571e-07,
 'NaOH': 1.3042049949307388e-07,
 'CaHCO3+': 0.004282339570472711,
 'NaHCO3': 0.0013114411706985558,
 'CO3--': 0.002723333000275801,
 'CaCO3': 0.006498382095290542,
 'NaCO3-': 0.0010233118248150646,
 'Na2CO3': 1.01447253598677e-05}
>>> solution.saturation_indexes
{'Calcite': 3.0673516648437325,
 'Aragonite': 2.923581817451497,
 'Vaterite': 2.500920170629433,
 'Halite': -4.1933510494726285}

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