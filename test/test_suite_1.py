# -*- coding: utf-8 -*-

import numpy as np
import phreeqpython

import pyequion2

temperature = 25
pp = phreeqpython.PhreeqPython()
solution_pp = pp.add_solution_simple({'CaCl2':1.0,'NaHCO3':2.0}, temperature = temperature)

eqsys = pyequion2.EquilibriumSystem(['CaCl2', 'NaHCO3'], activity_model="EXTENDED_DEBYE") #We set up the feed components of our system
molal_balance = {'Ca':0.001, 'C':0.002, 'Na':0.002, 'Cl':0.002} #Set up the balances
TK = 273.15 + temperature #Temperature in Kelvin
PATM = 1.0 #Pressure in atm
#Returns the solution class (the second argument are solver statistics, no need for understanding now)
solution, solution_stats = eqsys.solve_equilibrium_mixed_balance(TK, molal_balance=molal_balance, PATM=PATM)