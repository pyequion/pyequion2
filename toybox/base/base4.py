# -*- coding: utf-8 -*-

import pyequion

default_files = pyequion.builder.DEFAULT_DB_FILES
possible_reactions = pyequion.builder.get_all_possible_reactions(default_files)
possible_species = {k for reac in possible_reactions for k in reac.keys() if pyequion.builder._check_validity_specie_tag_in_reaction_dict(k)}