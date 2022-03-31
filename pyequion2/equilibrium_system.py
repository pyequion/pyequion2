# -*- coding: utf-8 -*-
import collections
import itertools

import numpy as np

from . import builder
from . import activity
from . import constants
from . import eqsolver
from . import solution
from . import fugacity
from . import logmaker
from . import sequencer


ACTIVITY_MODEL_MAP = {
    "IDEAL": activity.setup_ideal,
    "DEBYE": activity.setup_debye,
    "EXTENDED_DEBYE": activity.setup_extended_debye,
    "PITZER": activity.setup_pitzer,
}

class EquilibriumSystem():
    """
    Class for setting and calculate equilibria

    Attributes
    ----------
    base_species: List[str]
        Base aqueous species in system
    base_elements: List[str]
        Base elements in system
    species: List[str]
        Species in system
    reactions: List[dict]
        Reactions in system
    solid_reactions: List[dict]
        Solid Reactions in system
    formula_matrix: ndarray
        Formula matrix of system
    stoich_matrix: ndarray
        Stoichiometric matrix of system
    solid_formula_matrix: ndarray
        Solid formula matrix of system
    solid_stoich_matrix: ndarray
        Solid stoichiometric matrix of system
    activity_model: str
        Activity model for equilibrium
    calculate_water_activity: bool
        Whether to calculate water activity
    solverlog: None or str
        The log for last solver
    solvertype: None or str
        The type of last solver
    """

    def __init__(self, components, from_elements=False, activity_model="PITZER",
                 calculate_water_activity=False):
        """
        Parameters
        ----------
        components: List[str]
            Base components for defining reaction system.
        from_elements: bool
            If False, base components are species
            If True, base components are elements, and representative species
            are chosen for these elements (see builder.ELEMENT_SPECIES_MAP).
            Current elements implemented are
            C, Ca, Cl, Na, S, Ba, Mg, Fe, K, Sr,
            N, Cd, Li, Cu, Al, Br, F, Mn, P, Pb, Zn
        activity_model: str
            Model for activity coefficients. One of
            'IDEAL', 'DEBYE', 'EXTENDED_DEBYE', 'PITZER'
        calculate_water_activity: bool
            Whether to calculate water activity or assume it to be unit
        """
        self.base_species, self.base_elements = \
            _prepare_base(components, from_elements)
        self.species, self.reactions, self.solid_reactions, self.gas_reactions = \
            self._initialize_species_reactions()
        self.formula_matrix, self.stoich_matrix = \
            self._make_formula_and_stoich_matrices()
        self.solid_formula_matrix, self.solid_stoich_matrix = \
            self._make_solid_formula_and_stoich_matrices()
        self.gas_formula_matrix, self.gas_stoich_matrix = \
            self._make_gas_formula_and_stoich_matrices()
        self.activity_model = activity_model
        self.calculate_water_activity = calculate_water_activity
        self.solverlog = None
        self.solvertype = None
        self._activity_model_func = ACTIVITY_MODEL_MAP[activity_model](
            self.solutes, calculate_water_activity)
        self._fugacity_coefficient_function = lambda x, TK, P: 0.0
        self._x_molal = None
        self._x_act = None

    def set_activity_functions(self, activity_model="DEBYE",
                               calculate_water_activity=False):
        """
        Set activity model and function

        Parameters
        ----------
        activity_model: str
            One of ['IDEAL', 'DEBYE', 'EXTENDED_DEBYE', 'PITZER']
        """
        self.activity_model = activity_model
        self.calculate_water_activity = calculate_water_activity
        activity_setup = ACTIVITY_MODEL_MAP[activity_model]
        self._activity_model_func = activity_setup(self.solutes,
                                                   calculate_water_activity)

    def activity_function(self, molals, TK):
        """
        Activity function for aqueous species

        Parameters
        ----------
        molals: ndarray
            Molals of solutes
        TK: float
            Temperature in Kelvin

        Returns
        -------
        ndarray of activities (water is the first one, other solutes in molals order)
        """
        # molal to activities (including water)
        activity_model_res = self._activity_model_func(molals, TK)
        osmotic_coefficient, loggamma = \
            activity_model_res[0], activity_model_res[1:]
        if not self.calculate_water_activity:
            logact_water = 0.0
        else:
            logact_water = osmotic_coefficient * \
                constants.MOLAR_WEIGHT_WATER*np.sum(molals)
        logact_solutes = loggamma + np.log10(molals)
        logact = np.insert(logact_solutes, 0, logact_water)
        return logact

    def gas_activity_function(self, molals_gases, TK, P):
        """
        Activity function for gaseous species

        Parameters
        ----------
        molals_gases: np.ndarray
            Array of molals of gases
        TK: float
            Temperature in Kelvin
        P: float
            Pressure in atm

        Returns
        -------
        ndarray
            Log-activity of gases
        """
        # FIXME: Toy model. Just to make things work
        molal_fractions = molals_gases/np.sum(molals_gases)
        fugacity_coefficient_term = self._fugacity_coefficient_function(
            molal_fractions, TK, P)
        partial_pressure_term = np.log(P) + np.log(molal_fractions)
        logact = fugacity_coefficient_term + partial_pressure_term
        return logact

    def set_fugacity_coefficient_function(self, gas_indexes):
        """
        Parameters
        ----------
        gas_indexes: List[int]
            Indexes of gases to be considered

        Returns
        -------
        Callable[np.ndarray, float, float] -> float
            Log-fugacity function, accepting molal_fractions, temperature (TK), and pressure (ATM)
        """
        reactions_gases = [self.gas_reactions[i] for i in gas_indexes]
        if reactions_gases == []:  # Edge case
            self._fugacity_coefficient_function = lambda x, TK, P: 0.0
        else:
            self._fugacity_coefficient_function = \
                fugacity.make_peng_robinson_fugacity_function(reactions_gases)

    def get_log_equilibrium_constants(self, TK, PATM):
        """
        Parameters
        ----------
        TK: float
            Temperature in kelvins

        Returns
        -------
        List[float] of log equilibria constants of aqueous reactions
        """
        return builder.get_log_equilibrium_constants(self.reactions, TK, PATM)

    def get_solid_log_equilibrium_constants(self, TK, PATM):
        """
        Parameters
        ----------
        TK: float
            Temperature in kelvins

        Returns
        -------
        List[float] of log equilibria constants of solid reactions
        """
        return builder.get_log_equilibrium_constants(self.solid_reactions, TK, PATM)

    def get_gases_log_equilibrium_constants(self, TK, PATM):
        """
        Parameters
        ----------
        TK: float
            Temperature in kelvins

        Returns
        -------
        List[float] of log equilibria constants of gas reactions
        """
        return builder.get_log_equilibrium_constants(self.gas_reactions, TK, PATM)

    def solve_equilibrium_elements_balance(self, TK, element_balance,
                                           PATM=1.0,
                                           tol=1e-12,
                                           maxiter=1000,
                                           initial_guess='default'):
        """
        PARTIALLY DEPRECATED: Use solve_equilibrium_mixed_balance 
        except when explicitly calculating sequential balance equilibria
        
        Parameters
        ----------
        TK: float
            Temperature in Kelvins
        element_balance: dict[str, float]
            Dictionary of element balances
        tol: float
            Tolerance for solver
        initial_guess: ndarray or str
            Initial guess for solver. If 'default', is chosen as 0.1 for each specie

        Returns
        -------
        SolutionResult object of equilibrium solution and statistics of solver
        """
        assert len(element_balance) == self.nsolelements
        balance_vector = np.array([element_balance[el] for
                                   el in self.solute_elements])
        balance_vector = np.append(balance_vector, 0.0)
        balance_matrix = self.reduced_formula_matrix
        mask = 0
        balance_vector_log = np.zeros(0)
        balance_matrix_log = np.zeros((0, self.nspecies))
        mask_log = 0
        self.solverlog = logmaker.make_solver_log(element_balance,
                                                  dict(),
                                                  dict(),
                                                  dict(),
                                                  TK,
                                                  1.0,
                                                  "electroneutrality",
                                                  0.0)

        return self.solve_equilibrium_balance(balance_vector,
                                              balance_vector_log,
                                              balance_matrix,
                                              balance_matrix_log,
                                              mask,
                                              mask_log,
                                              TK,
                                              PATM,
                                              tol=tol,
                                              initial_guess=initial_guess)

    def solve_equilibrium_elements_balance_phases(self, TK, element_balance,
                                                  PATM=1.0,
                                                  solid_phases=None,
                                                  has_gas_phases=True,
                                                  tol=1e-12, maxiter=1000,
                                                  initial_guess='default'):
        """
        Parameters
        ----------
        TK: float
            Temperature in Kelvins
        element_balance: dict[str, float]
            Dictionary of element balances
        PATM: float
            Pressure in atms
        solid_phases: None or List[str]
            Phases of solid equilibria that precipitates. If None,
            we assume all stable-at-temperature TK phases precipitates
        has_gas_phases: bool
            Whether to consider gas equilibria
        tol: float
            Tolerance for solver
        initial_guess: ndarray or str or float
            Initial guess for solver. If 'default', is chosen as 0.1 for each specie.
            If float, is chosen as initial_guess for each specie

        Returns
        -------
        SolutionResult object of equilibrium solution and statistics of solver
        """
        balance_vector = np.array([element_balance[el] for
                                   el in self.solute_elements])
        balance_vector = np.append(balance_vector, 0.0)
        balance_matrix = self.reduced_formula_matrix
        if solid_phases is None:
            solid_phases = builder.get_most_stable_phases(
                self.solid_reactions, TK, PATM)
        if has_gas_phases:
            gas_phases = self.gas_phase_names
        else:
            gas_phases = []
        # FIXME: Remove need of popping H20
        if 'H2O(g)' in gas_phases:
            gas_phases.remove('H2O(g)')
        solid_indexes = self.get_solid_indexes(solid_phases)
        gas_indexes = self.get_gas_indexes(gas_phases)
        self.set_fugacity_coefficient_function(gas_indexes)

        activity_function = self.activity_function
        activity_function_gas = self.gas_activity_function

        balance_matrix_solids = self.reduced_solid_formula_matrix[:, solid_indexes]
        balance_matrix_gases = self.reduced_gas_formula_matrix[:, gas_indexes]
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK, PATM)
        log_solubility_constants = self.get_solid_log_equilibrium_constants(TK, PATM)
        log_solubility_constants = log_solubility_constants[solid_indexes]
        log_gases_constants = self.get_gases_log_equilibrium_constants(TK, PATM)
        log_gases_constants = log_gases_constants[gas_indexes]
        stoich_matrix = self.stoich_matrix
        stoich_matrix_solids = self.solid_stoich_matrix[solid_indexes, :]
        stoich_matrix_gases = self.gas_stoich_matrix[gas_indexes, :]

        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
            x_guess_solid = np.ones(len(solid_indexes))*0.1
            stability_solid_guess = np.zeros(len(solid_indexes))
            x_guess_gas = np.ones(len(gas_indexes))*0.1
            stability_gas_guess = np.zeros(len(gas_indexes))
        elif isinstance(initial_guess, float):
            x_guess = np.ones(self.nsolutes)*initial_guess
            x_guess_solid = np.ones(len(solid_indexes))*initial_guess
            stability_solid_guess = np.zeros(len(solid_indexes))
            x_guess_gas = np.ones(len(gas_indexes))*0.1
            stability_gas_guess = np.zeros(len(gas_indexes))
        else:
            x_guess, x_guess_solid, x_guess_gas, stability_solid_guess, stability_gas_guess = initial_guess

        molals, molals_solids, molals_gases, stability_solids, stability_gases, res = \
            eqsolver.solve_equilibrium_xlma_2(
                x_guess, x_guess_solid, x_guess_gas,
                stability_solid_guess, stability_gas_guess,
                TK, PATM, activity_function, activity_function_gas,
                balance_vector,
                log_equilibrium_constants, log_solubility_constants, log_gases_constants,
                balance_matrix, balance_matrix_solids, balance_matrix_gases,
                stoich_matrix, stoich_matrix_solids, stoich_matrix_gases,
                solver_function=None, tol=tol)
        self.solverlog = logmaker.make_solver_log(element_balance,
                                                  dict(),
                                                  dict(),
                                                  dict(),
                                                  TK,
                                                  1.0,
                                                  "electroneutrality",
                                                  0.0)
        if has_gas_phases:
            self.solvertype = "phase (with gas)"
        else:
            self.solvertype = "phase (no gas)"
        sol = solution.SolutionResult(self, molals, TK,
                                      molals_solids, solid_phases,
                                      molals_gases, gas_phases,
                                      PATM=PATM)
        stats = dict()
        stats['res'] = res
        stats['x'] = (molals, molals_solids, molals_gases, stability_solids, stability_gases)
        return sol, stats

    def solve_equilibrium_elements_balance_phases_sequential(self,
                                                             TK, element_balance,
                                                             PATM=1.0,
                                                             solid_phases=None,
                                                             has_gas_phases=True,
                                                             tol=1e-12, maxiter=1000,
                                                             initial_guess='default',
                                                             npoints=20):
        """
        Parameters
        ----------
        TK: float | (float, float)
            Temperature in Kelvins
        PATM: float
            Pressure in atms
        element_balance: dict[str, float | (float, float)]
            Dictionary of element balances
        solid_phases: None or List[str]
            Phases of solid equilibria that precipitates. If None,
            we assume all stable-at-temperature TK phases precipitates
        has_gas_phases: bool
            Whether to consider gas equilibria
        tol: float
            Tolerance for solver
        initial_guess: ndarray or str or float
            Initial guess for solver. If 'default', is chosen as 0.1 for each specie.
            If float, is chosen as initial_guess for each specie

        Returns
        -------
        list of SolutionResult object of equilibrium solution and list of residuals
        """
        TK_list, PATM_list, element_balance_list = \
            sequencer.transform_to_sequence_of_arguments(npoints, TK, PATM, element_balance)
        solutions = []
        residuals = []
        iterator = zip(TK_list, PATM_list, element_balance_list)
        for (TK, PATM, element_balance) in iterator:
            solution, stats = self.solve_equilibrium_elements_balance_phases(
                                              TK, element_balance,
                                              PATM,
                                              solid_phases,
                                              has_gas_phases,
                                              tol, maxiter,
                                              initial_guess)
            initial_guess = stats['x']
            solutions.append(solution)
            residuals.append(stats['res'])
        return solutions, residuals

    def solve_equilibrium_mixed_balance(self, TK, molal_balance=None,
                                        activities_balance=None,
                                        molal_balance_log=None,
                                        activities_balance_log=None,
                                        closing_equation='electroneutrality',
                                        closing_equation_value=0.0,
                                        PATM=1.0,
                                        tol=1e-12, maxiter=1000, initial_guess='default'):
        """
        Parameters
        ----------
        TK: float
            Temperature in Kelvins
        molal_balance: dict[str, float] or None
            Dictionary of molal balances
        activities_balance: dict[str, float] or None
            Dictionary of activities balances
        molal_balance_log: dict[str, float] or None
            Dictionary of log-molal balances
        activities_balance_log: dict[str, float] or None
            Dictionary of log-activities balances
        closing_equation: str or None
            Which closing equation to be used.
            If 'electroneutrality', closes assuming electroneutrality of elements
            If 'alkalinity', closes by alkaline balance defined by closing_equation_value
            If None, closure must come from dictionaries
        closing_equation_value: float
            Value of closing equation
        PATM: float
            Pressure in atms
        tol: float
            Tolerance for solver
        maxiter: int
            Maximum iterations for solver
        initial_guess: ndarray or str
            Initial guess for solver. If 'default', is chosen as 0.1 for each specie

        Returns
        -------
        SolutionResult object of equilibrium solution and statistics of solver

        """
        
        molal_balance = _none_to_dict(molal_balance)
        activities_balance = _none_to_dict(activities_balance)
        molal_balance_log = _none_to_dict(molal_balance_log)
        activities_balance_log = _none_to_dict(activities_balance_log)
        # TODO : Bunch of assertions
        balance = _join_and_tag_dicts(molal_balance, activities_balance)
        balance_log = _join_and_tag_dicts(
            molal_balance_log, activities_balance_log)
        balance_vector, balance_matrix, mask = \
            self._prepare_balance_arrays(balance)
        balance_vector_log, balance_matrix_log, mask_log = \
            self._prepare_balance_arrays(balance_log)
        if closing_equation:
            balance_vector = np.append(balance_vector, closing_equation_value)
            if closing_equation == 'electroneutrality':
                closing_row = self.charge_vector
                closing_mask = 0
            elif closing_equation == 'alkalinity':
                closing_row = self.alkalinity_vector
                closing_mask = 1
            balance_matrix = np.vstack([balance_matrix, closing_row])
            mask = np.hstack([mask, closing_mask])
        self.solverlog = logmaker.make_solver_log(molal_balance,
                                                  activities_balance,
                                                  molal_balance_log,
                                                  activities_balance_log,
                                                  TK,
                                                  PATM,
                                                  closing_equation,
                                                  closing_equation_value)
        self.solvertype = "aqueous"
        return self.solve_equilibrium_balance(balance_vector,
                                              balance_vector_log,
                                              balance_matrix,
                                              balance_matrix_log,
                                              mask,
                                              mask_log,
                                              TK,
                                              PATM,
                                              tol=tol,
                                              initial_guess=initial_guess)

    def solve_equilibrium_mixed_balance_sequential(self, TK, molal_balance=None,
                                                   activities_balance=None,
                                                   molal_balance_log=None,
                                                   activities_balance_log=None,
                                                   closing_equation='electroneutrality',
                                                   closing_equation_value=0.0,
                                                   PATM=1.0,
                                                   tol=1e-12, maxiter=1000,
                                                   initial_guess='default',
                                                   npoints=20):
        """
        Parameters
        ----------
        TK: float | (float, float)
            Temperature in Kelvins
        molal_balance: dict[str, float | (float, float)] or None
            Dictionary of molal balances
        activities_balance: dict[str, float | (float, float)] or None
            Dictionary of activities balances
        molal_balance_log: dict[str, float | (float, float)] or None
            Dictionary of log-molal balances
        activities_balance_log: dict[str, float | (float, float)] or None
            Dictionary of log-activities balances
        closing_equation: str or None
            Which closing equation to be used.
            If 'electroneutrality', closes assuming electroneutrality of elements
            If 'alkalinity', closes by alkaline balance defined by closing_equation_value
            If None, closure must come from dictionaries
        closing_equation_value: float | (float, float)
            Value of closing equation
        PATM: float
            Pressure in atms
        tol: float
            Tolerance for solver
        maxiter: int
            Maximum iterations for solver
        initial_guess: ndarray or str
            Initial guess for solver. If 'default', is chosen as 0.1 for each specie

        Returns
        -------
        list of SolutionResult object of equilibrium solution and list of residuals

        """
        molal_balance = _none_to_dict(molal_balance)
        activities_balance = _none_to_dict(activities_balance)
        molal_balance_log = _none_to_dict(molal_balance_log)
        activities_balance_log = _none_to_dict(activities_balance_log)
        iterator = zip(*sequencer.transform_to_sequence_of_arguments(
                        npoints, TK, molal_balance, activities_balance, 
                        molal_balance_log, activities_balance_log, 
                        closing_equation_value, PATM))
        solutions = []
        residuals = []
        for (TK, molal_balance, activities_balance, molal_balance_log, 
             activities_balance_log, closing_equation_value, PATM) in iterator:
            
            solution, stats = \
                self.solve_equilibrium_mixed_balance(TK,
                                                     molal_balance,
                                                     activities_balance,
                                                     molal_balance_log,
                                                     activities_balance_log,
                                                     closing_equation,
                                                     closing_equation_value,
                                                     PATM,
                                                     tol, maxiter,
                                                     initial_guess)
            initial_guess = stats['x']
            solutions.append(solution)
            residuals.append(stats['res'])
        return solutions, stats
    
    def solve_equilibrium_balance(self,
                                  balance_vector,
                                  balance_vector_log,
                                  balance_matrix,
                                  balance_matrix_log,
                                  mask,
                                  mask_log,
                                  TK,
                                  PATM=1.0,
                                  tol=1e-12, maxiter=1000,
                                  initial_guess='default'):
        """
        Parameters
        ----------
        balance_vector: ndarray
            Vector of balances
        balance_vector_log: ndarray
            Vector of log-balances
        balance_matrix: ndarray
            Matrix of balances
        balance_matrix_log: ndarray
            Matrix of log-balances
        mask: ndarray
            Activity mask
        mask_log: ndarray
            Log-activity mask
        TK: float
            Temperature in Kelvin
        tol: float
            Tolerance of solver
        initial_guess: ndarray or str
            Initial guess for solver. If 'default', is chosen as 0.1 for each specie
        Returns
        -------
        SolutionResult object of equilibrium solution and statistics of solver

        """
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK, PATM)
        stoich_matrix = self.stoich_matrix
        activity_function = self.activity_function
        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
        elif isinstance(initial_guess, float):
            x_guess = np.ones(self.nsolutes)*initial_guess
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
        stats = dict()
        stats['res'] = res
        stats['x'] = x
        return solution.SolutionResult(self, x, TK, PATM=PATM), stats

    @property
    def elements(self):
        """Alias for base elements"""
        return self.base_elements

    @property
    def extended_elements(self):
        """Elements + 'e'"""
        return self.elements + ['e']

    @property
    def solute_elements(self):  # Ignore H and O
        """Elements excluding H and O"""
        return self.elements[2:]

    @property
    def solutes(self):
        """Solutes"""
        return self.species[1:]  # Ignore water

    @property
    def nspecies(self):
        """Number of aqueous species (includes H2O)"""
        return len(self.species)

    @property
    def nreactions(self):
        """Number of aqueous reactions"""
        return len(self.reactions)

    @property
    def nelements(self):
        """Number of elements"""
        return len(self.elements)

    @property
    def nsolelements(self):
        """Number of solute elements"""
        return len(self.solute_elements)

    @property
    def nsolutes(self):
        """Number of solutes"""
        return len(self.solutes)

    @property
    def reduced_formula_matrix(self):
        """Formula matrix excluding H and O elements"""
        return self.formula_matrix[2:, :]

    @property
    def reduced_solid_formula_matrix(self):
        """Solid formula matrix excluding H and O elements"""
        return self.solid_formula_matrix[2:, :]

    @property
    def reduced_gas_formula_matrix(self):
        """Solid formula matrix excluding H and O elements"""
        return self.gas_formula_matrix[2:, :]

    @property
    def charge_vector(self):
        """Vector of charge number for aqueous species"""
        return self.formula_matrix[-1, :]

    @property
    def solutes_charge_vector(self):
        """Vector of charge numbers for solutes"""
        return self.charge_vector[1:]

    @property
    def alkalinity_vector(self):
        """Vector of alkalinity coefficients"""
        return np.array([constants.ALKALINE_COEFFICIENTS.get(specie, 0.0)
                         for specie in self.species])

    @property
    def solutes_alkalinity_vector(self):
        """Vector of alkalinity coefficients for solutes"""
        return self.alkalinity_vector[1:]

    @property
    def solid_phase_names(self):
        """Names of solid phases"""
        return [sol_reac['phase_name'] for sol_reac in self.solid_reactions]

    @property
    def gas_phase_names(self):
        """Names of gas phases"""
        return [gas_reac["phase_name"] for gas_reac in self.gas_reactions]

    def _initialize_species_reactions(self):
        return builder.get_species_reaction_from_initial_species(
            self.base_species)

    def _make_formula_and_stoich_matrices(self):
        formula_matrix = builder.make_formula_matrix(
            self.species, self.elements)
        stoich_matrix = builder.make_stoich_matrix(
            self.species, self.reactions)
        return formula_matrix, stoich_matrix

    def _make_solid_formula_and_stoich_matrices(self):
        if not self.solid_reactions:  # No precipitating solid phases
            solid_formula_matrix = np.zeros((self.nelements+1, 0))
            solid_stoich_matrix = np.zeros((0, self.nspecies))
        else:
            solid_formula_matrix = builder.make_solid_formula_matrix(
                self.solid_reactions, self.elements)
            solid_stoich_matrix = builder.make_stoich_matrix(
                self.species, self.solid_reactions)
        return solid_formula_matrix, solid_stoich_matrix

    def _make_gas_formula_and_stoich_matrices(self):
        if not self.gas_reactions:
            gas_formula_matrix = np.zeros((self.nelements+1, 0))
            gas_stoich_matrix = np.zeros((0, self.nspecies))
        else:
            gas_formula_matrix = builder.make_gas_formula_matrix(
                self.gas_reactions, self.elements)
            gas_stoich_matrix = builder.make_stoich_matrix(
                self.species, self.gas_reactions)
        return gas_formula_matrix, gas_stoich_matrix

    def _prepare_balance_arrays(self, balances):
        nbalances = len(balances)
        balance_vector = np.zeros(nbalances)
        mask = np.zeros(nbalances)
        balance_matrix = np.zeros((nbalances, self.nspecies))
        for i, (key, (value, tag)) in enumerate(balances.items()):
            balance_vector[i] = value
            mask[i] = tag
            if key in self.elements:
                balance_matrix[i, :] = self.formula_matrix[self.elements.index(
                    key), :]
            elif key in self.species:
                balance_matrix[i, self.species.index(key)] = 1.0
        return balance_vector, balance_matrix, mask

    def get_solid_indexes(self, solid_phases):
        indexes = [None for _ in range(len(solid_phases))]
        for i, solid_phase in enumerate(solid_phases):
            for j, solid_reaction in enumerate(self.solid_reactions):
                if solid_reaction['phase_name'] == solid_phase:
                    indexes[i] = j
        return indexes

    def get_gas_indexes(self, gas_phases):
        indexes = [None for _ in range(len(gas_phases))]
        for i, gas_phase in enumerate(gas_phases):
            for j, gas_reaction in enumerate(self.gas_reactions):
                if gas_reaction['phase_name'] == gas_phase:
                    indexes[i] = j
        return indexes


# Helpers
def _prepare_base(components, from_elements):
    # Implicity assumes that O and H will always in elements,
    # and H2O will always be a species (were talking about aqueous
    # equilibrium, after all).
    if from_elements:
        base_elements = set(components)
        base_elements.add('O')
        base_elements.add('H')
        base_species = builder.elements_to_species(base_elements)
    else:
        base_species = set(components)
        base_species.add('H2O')
        base_elements = builder.species_to_elements(base_species)
    # Work back to list
    base_elements = builder.set_h_and_o_as_first_elements(
        list(base_elements))
    base_species = builder.set_h2o_as_first_specie(list(base_species))
    return base_species, base_elements


def _none_to_dict(d):
    return d if d is not None else dict()


def _join_and_tag_dicts(*args):
    tags = list(range(len(args)))
    d = dict()
    for i, di in enumerate(args):
        d.update({k: (v, tags[i]) for k, v in di.items()})
    return d