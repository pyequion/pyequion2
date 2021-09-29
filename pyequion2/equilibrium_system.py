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
        self.solid_formula_matrix, self.solid_stoich_matrix = \
            self.make_solid_formula_and_stoich_matrices()
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
    
    def make_solid_formula_and_stoich_matrices(self):
        solid_formula_matrix = builder.make_solid_formula_matrix(self.solid_reactions, self.elements)
        solid_stoich_matrix = builder.make_stoich_matrix(self.species, self.solid_reactions)
        return solid_formula_matrix, solid_stoich_matrix

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
        
    def solve_equilibrium_elements_balance(self, TK, element_balance,
                                           tol=1e-6, initial_guess='default',
                                           return_solution=True):
        assert(len(element_balance) == self.nsolelements)
        balance_vector = np.array([element_balance[el] for 
                                    el in self.solute_elements])
        balance_vector = np.append(balance_vector,0.0)
        balance_matrix = self.reduced_formula_matrix
        mask = 0
        balance_vector_log = np.zeros(0)
        balance_matrix_log = np.zeros((0, self.nspecies))
        mask_log = 0
        return self.solve_equilibrium_balance(balance_vector,
                                              balance_vector_log,
                                              balance_matrix,
                                              balance_matrix_log,
                                              mask, 
                                              mask_log,
                                              TK,
                                              tol=tol,
                                              initial_guess=initial_guess,
                                              return_solution=return_solution)
    
    def solve_equilibrium_elements_balance_solids(self, TK, element_balance, 
                                                  solid_phases=None,
                                                  tol=1e-6,
                                                  initial_guesses='default',
                                                  return_solution=True):
        balance_vector = np.array([element_balance[el] for 
                                    el in self.solute_elements])
        balance_vector = np.append(balance_vector,0.0)
        balance_matrix = self.reduced_formula_matrix
        mask = 0
        balance_vector_log = np.zeros(0)
        balance_matrix_log = np.zeros((0, self.nspecies))
        mask_log = 0
        if solid_phases is None:
            solid_phases = builder.get_most_stable_phases(self.solid_reactions, TK)
        solid_indexes = self._get_solid_indexes(solid_phases)
        balance_matrix_p = self.reduced_solid_formula_matrix[:, solid_indexes]
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK)
        log_solubility_constants = self.get_solid_log_equilibrium_constants(TK)
        log_solubility_constants = log_solubility_constants[solid_indexes]
        stoich_matrix = self.stoich_matrix
        activity_function = self.activity_function
        stoich_matrix_p = self.solid_stoich_matrix[solid_indexes, :]
        
        if initial_guesses == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
            x_guess_p = np.ones(len(solid_indexes))*0.1
            stability_guess_p = np.zeros(len(solid_indexes))
        else:
            x_guess, x_guess_p, stability_guess_p = np.array(initial_guesses)
        molals, molals_p, stability_indexes_p, res = \
                    eqsolver.solve_equilibrium_xlma(
                           x_guess, x_guess_p, stability_guess_p,
                           TK, activity_function,
                           balance_vector, balance_vector_log,
                           log_equilibrium_constants, log_solubility_constants,
                           balance_matrix, balance_matrix_log, balance_matrix_p,
                           stoich_matrix, stoich_matrix_p,
                           mask, mask_log,
                           solver_function=None,
                           tol=1e-6)
        if return_solution:
            sol = solution.SolutionResult(self, molals, TK,
                                          molals_p, solid_phases)
            return sol, res
        else:
            return molals, res
    
    def solve_equilibrium_mixed_balance(self, TK, molal_balance=None,
                                        activities_balance=None,
                                        molal_balance_log=None,
                                        activities_balance_log=None,
                                        closing_equation='electroneutrality',
                                        closing_equation_value=0.0,
                                        tol=1e-6, initial_guess='default',
                                        return_solution=True):
        molal_balance = _none_to_dict(molal_balance)
        activities_balance = _none_to_dict(activities_balance)
        molal_balance_log = _none_to_dict(molal_balance_log)
        activities_balance_log = _none_to_dict(activities_balance_log)
        #TODO : Bunch of assertions
        balance = _join_and_tag_dicts(molal_balance, activities_balance)
        balance_log = _join_and_tag_dicts(molal_balance_log, activities_balance_log)
        balance_vector, balance_matrix, mask = \
            self._prepare_balance_arrays(balance)
        balance_vector_log, balance_matrix_log, mask_log = \
            self._prepare_balance_arrays(balance_log)
        if closing_equation:
            balance_vector = np.append(balance_vector, closing_equation_value)
            if closing_equation == 'electroneutrality':
                closing_row = self.charge_vector
            elif closing_equation == 'alkalinity':
                closing_row = self.alkalinity_vector
            balance_matrix = np.vstack([balance_matrix, closing_row])
        return self.solve_equilibrium_balance(balance_vector,
                                              balance_vector_log,
                                              balance_matrix,
                                              balance_matrix_log,
                                              mask, 
                                              mask_log,
                                              TK,
                                              initial_guess=initial_guess,
                                              return_solution=return_solution)
    
    def solve_equilibrium_balance(self,
                                  balance_vector,
                                  balance_vector_log,
                                  balance_matrix,
                                  balance_matrix_log,
                                  mask, 
                                  mask_log,
                                  TK,
                                  tol=1e-6,
                                  initial_guess='default',
                                  return_solution=True):
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK)
        stoich_matrix = self.stoich_matrix
        activity_function = self.activity_function
        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
        else:
            x_guess = np.array(initial_guess)
        x, res = eqsolver.solve_equilibrium_solutes(
                              x_guess,
                              TK,
                              activity_function,
                              balance_vector,
                              balance_vector_log,
                              log_equilibrium_constants,
                              balance_matrix,
                              balance_matrix_log,
                              stoich_matrix,
                              mask, 
                              mask_log,
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
    def reduced_solid_formula_matrix(self):
        return self.solid_formula_matrix[2:, :]

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
    
    
    def _prepare_balance_arrays(self, balances):
        nbalances = len(balances)
        balance_vector = np.zeros(nbalances)
        mask = np.zeros(nbalances)
        balance_matrix = np.zeros((nbalances, self.nspecies))
        for i, (key, (value, tag)) in enumerate(balances.items()):
            balance_vector[i] = value
            mask[i] = tag
            if key in self.elements:
                balance_matrix[i, :] = self.formula_matrix[self.elements.index(key), :]
            elif key in self.species:
                balance_matrix[i, self.species.index(key)] = 1.0
        return balance_vector, balance_matrix
    
    def _get_solid_indexes(self, solid_phases):
        indexes = [None for _ in range(len(solid_phases))]
        for i, solid_phase in enumerate(solid_phases):
            for j, solid_reaction in enumerate(self.solid_reactions):
                if solid_reaction['phase_name'] == solid_phase:
                    indexes[i] = j
        return indexes


#Helpers
def _none_to_dict(d):
    return d if d is not None else dict()


def _join_and_tag_dicts(*args):
    tags = list(range(len(args)))
    d = dict()
    for i, di in enumerate(args):
        d.update({k: (v, tags[i]) for k, v in di.items()})
    return d