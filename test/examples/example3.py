# -*- coding: utf-8 -*-

from pyequion2 import EquilibriumSystem #Import the necessary module
eqsys = EquilibriumSystem(['C', 'Na', 'Ca', 'Cl'], from_elements=True) #We set up the feed components of our system
molal_balance = {'C':0.075, 'Na':0.075, 'Cl':0.056, 'Ca':0.028} #Set up the balances
TK = 298.15 #Temperature in Kelvin
PATM = 1e0 #Pressure in atm
#Returns the solution class (the second argument are solver statistics, no need for understanding now)
solution, solution_stats = eqsys.solve_equilibrium_elements_balance_phases(TK,
                                                                           molal_balance,
                                                                           solid_phases=[],
                                                                           has_gas_phases=True,
                                                                           PATM=PATM)
print(solution.gas_molals)

#log K + log CO2(g) = log CO2
