# -*- coding: utf-8 -*-
import numpy as np

from . import builder
from . import activity
from . import constants
from . import eqsolver

ACTIVITY_MODEL_MAP = {
        "IDEAL":activity.setup_ideal,
        "DEBYE":activity.setup_debyehuckel,
        "PITZER":activity.setup_pitzer,
        }


class EquilibriumSystem():
    def __init__(self, components, from_elements=False):
        self.base_species, self.base_elements = \
            self.prepare_base(components, from_elements)
        self.species, self.reactions = self.initialize_species_reactions()
        self.formula_matrix, self.stoich_matrix = \
            self.make_formula_and_stoich_matrices()
        self.activity_model = None
        self.equilibrium_molals = None
        
    def prepare_base(self, components, from_elements):
        #Implicity assumes that O and H will always in elements,
        #and H2O will always be a species (were talking about aqueous
        #equilibrium, after all).
        if from_elements:
            base_elements = set(components)
            base_elements.add('O')
            base_elements.add('H')
            base_species = builder.elements_to_species(base_elements)
        else:
            base_species = set(components)
            base_species.add('H2O')
            base_elements = builder.species_to_elements(base_species)
        #Work back to list
        base_elements = builder.set_h_and_o_as_first_elements(list(base_elements))
        base_species = builder.set_h2o_as_first_specie(list(base_species))
        return base_species, base_elements

    def initialize_species_reactions(self):
        return builder.get_species_reaction_from_initial_species(
                self.base_species)

    def make_formula_and_stoich_matrices(self):
        formula_matrix = builder.make_formula_matrix(self.species, self.elements)
        stoich_matrix = builder.make_stoich_matrix(self.species, self.reactions)
        return formula_matrix, stoich_matrix

    def set_activity_function(self, activity_model="DEBYE"):
        activity_setup = ACTIVITY_MODEL_MAP[activity_model]
        self.activity_model = activity_setup(self.solutes)

    def activity_function(self,molals,TK):
        #molal to activities (including water)
        activity_model_res = self.activity_model(molals,TK)
        osmotic_pressure, loggamma = \
            activity_model_res[0], activity_model_res[1:]
        logact_water = osmotic_pressure*constants.MOLAR_WEIGHT_WATER*np.sum(molals)
        logact_solutes = loggamma + np.log10(molals)
        logact = np.insert(logact_solutes, 0, logact_water)
        return logact

    def get_log_equilibrium_constants(self, TK):
        return builder.get_log_equilibrium_constants(self.reactions, TK)
    
    def solve_equilibrium(self, molal_balances, TK,
                          tol=1e-6, initial_guess='default'):
        formula_vector_ = np.array([molal_balances[el] for el in self.elements])
        formula_vector = np.insert(formula_vector_,0,0.0)
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK)
        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.001
        else:
            x_guess = np.array(initial_guess)
        x, res = eqsolver.solve_equilibrium_solutes(
                              x_guess,
                              TK,
                              self.activity_function,
                              formula_vector,
                              self.reduced_formula_matrix,
                              self.stoich_matrix,
                              log_equilibrium_constants)
        self.equilibrium_molals = x
        return x, res
    
    @property
    def elements(self):
        return self.base_elements

    @property
    def solute_elements(self): #Ignore H and O
        return self.elements[2:]

    @property
    def solutes(self):
        return self.species[1:] #Ignore water

    @property
    def nspecies(self):
        return len(self.species)

    @property
    def nreactions(self):
        return len(self.reactions)

    @property
    def nelements(self):
        return len(self.elements)

    @property
    def nsolelements(self):
        return len(self.solute_elements)

    @property
    def nsolutes(self):
        return len(self.solutes)

    @property
    def reduced_formula_matrix(self):
        return self.formula_matrix[2:, :]