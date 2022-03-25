# -*- coding: utf-8 -*-

from pyequion2 import EquilibriumSystem #Import the necessary module
eqsys = EquilibriumSystem(['Na', 'C'], from_elements=True) #We set up the feed components of our system
molal_balance = {'Na': 0.150} #Set up the balances
activities_balance_log = {'H+': -9.0}
TK = 35 + 273.15 #Temperature in Kelvin
solution, solution_stats = eqsys.solve_equilibrium_mixed_balance(TK,
                                                                 molal_balance=molal_balance,
                                                                 activities_balance_log=activities_balance_log)
