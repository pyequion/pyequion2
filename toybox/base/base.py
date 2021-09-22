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
print(x)
print(eqsys.activity_model(x,298.15))
print(eqsys.activity_function(x,298.15))
#print(eqsys.base_species)
#print(builder._get_elements_and_their_coefs(eqsys.base_species))
#print(builder.get_species_reaction_from_initial_species(eqsys.base_species))