# -*- coding: utf-8 -*-

from pyequion2 import EquilibriumSystem #Import the necessary module
eqsys = EquilibriumSystem(['Ca', 'Na', 'C', 'Cl'], from_elements=True) #We set up the feed components of our system
molal_balance = {'Ca':0.028, 'C':0.065, 'Na':0.065, 'Cl':0.056} #Set up the balances
TK = 298.15 #Temperature in Kelvin
PATM = 1.0 #Pressure in atm
#Returns the solution class (the second argument are solver statistics, no need for understanding now)
solution, solution_stats = eqsys.solve_equilibrium_elements_balance_phases(TK,
                                                                           molal_balance)
