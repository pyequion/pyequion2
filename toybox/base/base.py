# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../..')

import numpy as np

from pyequion2 import EquilibriumSystem
from pyequion2 import builder
from pyequion2 import converters

eqsys = EquilibriumSystem(['C','Ca','Na','Cl'], from_elements=True)
eqsys.set_activity_function("DEBYE")
c = np.ones(eqsys.nsolutes)*10
x = converters.mmolar_to_molal(c)
y = eqsys.activity_function(x,298.15)

molal_balances = {'Ca':0.028, 'Cl':0.056, 'Na':0.075, 'C':0.065}
TK = 298.15
solution,res = eqsys.solve_equilibrium(molal_balances, TK, tol=1e-12)
print(res)
print(solution.equilibrium_molals)
print(solution.equilibrium_activities)
print(solution.equilibrium_concentrations)
 #print(eqsys.base_species)
#print(builder._get_elements_and_their_coefs(eqsys.base_species))
#print(builder.get_species_reaction_from_initial_species(eqsys.base_species))