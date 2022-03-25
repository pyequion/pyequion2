# -*- coding: utf-8 -*-

from pyequion2 import EquilibriumSystem #Import the necessary module
eqsys = EquilibriumSystem(['CO2']) #We set up the feed components of our system
molal_balance = {'C':0.5} #Set up the balances
TK = 298.15
solution, _ = eqsys.solve_equilibrium_elements_balance_phases(TK,
                                                              molal_balance)
PATM = 100.0
solution_high_pressure, _ = eqsys.solve_equilibrium_elements_balance_phases(TK,
                                                                            molal_balance,
                                                                            PATM=PATM)
