# -*- coding: utf-8 -*-

import pyequion2
eqsys = pyequion2.EquilibriumSystem(['CaCl2', 'NaHCO3']) #We set up the feed components of our system
molal_balance = {'Ca':0.028, 'C':0.065, 'Na':0.065, 'Cl':0.056} #Set up the balances
TK = 298.15 #Temperature in Kelvin
PATM = 1.0 #Pressure in atm
#Returns the solution class (the second argument are solver statistics, no need for understanding now)
solution, solution_stats = eqsys.solve_equilibrium_mixed_balance(TK, molal_balance=molal_balance, PATM=PATM)