# -*- coding: utf-8 -*-
import numpy as np

from . import builder
from . import activity
from . import constants
from . import eqsolver
from . import solution


ACTIVITY_MODEL_MAP = {
        "IDEAL":activity.setup_ideal,
        "DEBYE":activity.setup_debyehuckel,
        "PITZER":activity.setup_pitzer,
        }


class EquilibriumSystem():
    def __init__(self, components, from_elements=False, activity_model="DEBYE"):
        self.base_species, self.base_elements = \
            self.prepare_base(components, from_elements)
        self.species, self.reactions, self.solid_reactions = \
            self.initialize_species_reactions()
        self.formula_matrix, self.stoich_matrix = \
            self.make_formula_and_stoich_matrices()
        self.activity_model = ACTIVITY_MODEL_MAP[activity_model](self.solutes)
        self.TK = None
        self._x_molal = None
        self._x_act = None
        
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
    
    def get_solid_log_equilibrium_constants(self, TK):
        return builder.get_log_equilibrium_constants(self.solid_reactions, TK)
        
    def solve_equilibrium_elements_balance(self, element_balance, TK,
                                           tol=1e-6, initial_guess='default',
                                           return_solution=True):
        assert(len(element_balance) == self.nsolelements)
        balance_vector_ = np.array([element_balance[el] for 
                                    el in self.solute_elements])
        balance_vector = np.append(balance_vector_,0.0)
        balance_matrix = self.reduced_formula_matrix
        return self.solve_equilibrium_balance(balance_matrix,
                                              balance_vector,
                                              TK,
                                              tol=tol,
                                              initial_guess=initial_guess,
                                              return_solution=return_solution)

    def solve_equilibrium_species_balance(self, species_balance, TK,
                                          tol=1e-6, initial_guess='default',
                                          return_solution=True):
        assert(len(species_balance) == self.nsolelements)
        balance_matrix_1 = np.zeros((self.nsolelements, self.nspecies))
        balance_vector_ = np.zeros(self.nsolelements)
        for i,(key,value) in enumerate(species_balance.items()):
            balance_vector_[i] = value
            balance_matrix_1[i, self.species.index(key)] = 1.0
        balance_vector = np.append(balance_vector_,0.0)
        balance_matrix = np.vstack([balance_matrix_1, self.charge_vector])
        return self.solve_equilibrium_balance(balance_matrix,
                                              balance_vector,
                                              TK,
                                              tol=tol,
                                              initial_guess=initial_guess,
                                              return_solution=return_solution)

    def solve_equilibrium_mixed_balance(self, mixed_balance, TK,
                                          tol=1e-6, initial_guess='default',
                                          return_solution=True):
        assert(len(mixed_balance) == self.nsolelements)
        balance_matrix_1 = np.zeros((self.nsolelements, self.nspecies))
        balance_vector_ = np.zeros(self.nsolelements)
        for i,(key,value) in enumerate(mixed_balance.items()):
            balance_vector_[i] = value
            if key in self.elements:
                balance_matrix_1[i, :] = \
                    self.formula_matrix[self.elements.index(key), :]
            elif key in self.species:
                balance_matrix_1[i, self.species.index(key)] = 1.0
            else:
                raise ValueError("Key of mixed_balance neither element nor specie in model")
        balance_vector = np.append(balance_vector_,0.0)
        balance_matrix = np.vstack([balance_matrix_1, self.charge_vector])
        return self.solve_equilibrium_balance(balance_matrix,
                                              balance_vector,
                                              TK,
                                              tol=tol,
                                              initial_guess=initial_guess,
                                              return_solution=return_solution)

    def solve_equilibrium_balance_alkalinity(self, mixed_balance, TK, alkalinity,
                                             tol=1e-6, initial_guess='default',
                                             return_solution=True):
        assert(len(mixed_balance) == self.nsolelements)
        balance_matrix_1 = np.zeros((self.nsolelements, self.nspecies))
        balance_vector_ = np.zeros(self.nsolelements)
        for i,(key,value) in enumerate(mixed_balance.items()):
            balance_vector_[i] = value
            if key in self.elements:
                balance_matrix_1[i, :] = \
                    self.formula_matrix[self.elements.index(key), :]
            elif key in self.species:
                balance_matrix_1[i, self.species.index(key)] = 1.0
            else:
                raise ValueError("Key of mixed_balance neither element nor specie in model")
        balance_vector = np.append(balance_vector_, alkalinity)
        balance_matrix = np.vstack([balance_matrix_1, self.alkalinity_vector])
        return self.solve_equilibrium_balance(balance_matrix,
                                              balance_vector,
                                              TK,
                                              tol=tol,
                                              initial_guess=initial_guess,
                                              return_solution=return_solution)

    def solve_equilibrium_balance(self, balance_matrix, balance_vector, TK,
                                  tol=1e-6, initial_guess='default',
                                  return_solution=True):
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK)
        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
        else:
            x_guess = np.array(initial_guess)
        x, res = eqsolver.solve_equilibrium_solutes(
                              x_guess,
                              TK,
                              self.activity_function,
                              balance_vector,
                              balance_matrix,
                              self.stoich_matrix,
                              log_equilibrium_constants,
                              tol=tol)
        if return_solution:
            return solution.SolutionResult(self, x, TK), res
        else:
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

    @property
    def charge_vector(self):
        return self.formula_matrix[-1,:]

    @property
    def solutes_charge_vector(self):
        return self.charge_vector[1:]

    @property
    def alkalinity_vector(self):
        return np.array([constants.ALKALINE_COEFFICIENTS.get(specie,0.0)
                         for specie in self.species])
    
    @property
    def solutes_alkalinity_vector(self):
        return self.alkalinity_vector[1:]