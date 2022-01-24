# -*- coding: utf-8 -*-

import pyequion

default_files = pyequion.builder.DEFAULT_DB_FILES
possible_reactions = pyequion.builder.get_all_possible_reactions(default_files)
species, reactions = \
    pyequion.builder._get_species_reactions_from_compounds(['Al+++', 'H2O'], possible_reactions)
print(species, reactions)

