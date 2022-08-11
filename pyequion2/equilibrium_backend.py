# -*- coding: utf-8 -*-
import collections
import itertools
import warnings

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except (ImportError, AssertionError):
    warnings.warn("JAX not installed. Only numpy can be chosen as backend")

from . import builder
from . import activity
from . import constants
from . import fugacity
from .interface import diffusion_coefficients


ACTIVITY_MODEL_MAP = {
    "IDEAL": activity.setup_ideal,
    "DEBYE_LIMITING": activity.setup_debye,
    "DEBYE": activity.setup_extended_debye,
    "EXTENDED_DEBYE": activity.setup_extended_debye,
    "PITZER": activity.setup_pitzer,
}


class EquilibriumBackend():
    """
    Class for getting equilibria calculation, for off-the-shelf computation

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

    def __init__(self, components, from_elements=False, activity_model="EXTENDED_DEBYE",
                 calculate_water_activity=False, backend='numpy'):
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
        self.backend = backend
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
        self.activity_model_func = ACTIVITY_MODEL_MAP[activity_model](
            self.solutes, calculate_water_activity, self.backend)
        self._fugacity_coefficient_function = lambda x, TK, P: 0.0
        self._x_molal = None
        self._x_act = None

    def update_system(self,
                      possible_reactions=None,
                      possible_solid_reactions=None,
                      possible_gas_reactions=None):
        self.species, self.reactions, self.solid_reactions, self.gas_reactions = \
            self._initialize_species_reactions(possible_reactions,
                                               possible_solid_reactions,
                                               possible_gas_reactions)
        self.formula_matrix, self.stoich_matrix = \
            self._make_formula_and_stoich_matrices()
        self.solid_formula_matrix, self.solid_stoich_matrix = \
            self._make_solid_formula_and_stoich_matrices()
        self.gas_formula_matrix, self.gas_stoich_matrix = \
            self._make_gas_formula_and_stoich_matrices()
        self.solverlog = None
        self.solvertype = None
        self.activity_model_func = ACTIVITY_MODEL_MAP[self.activity_model](
            self.solutes, self.calculate_water_activity)
        self._fugacity_coefficient_function = lambda x, TK, P: 0.0
        self._x_molal = None
        self._x_act = None
    
    def set_activity_functions(self, activity_model="EXTENDED_DEBYE",
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
        self.activity_model_func = activity_setup(self.solutes,
                                                   calculate_water_activity,
                                                   self.backend)

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
        activity_model_res = self.activity_model_func(molals, TK)
        osmotic_coefficient, loggamma = \
            activity_model_res[0], activity_model_res[1:]
        if not self.calculate_water_activity:
            logact_water = 0.0
        else:
            logact_water = osmotic_coefficient * \
                constants.MOLAR_WEIGHT_WATER*self._np.sum(molals)
        logact_solutes = loggamma + self._np.log10(molals)
        logact = self._np.insert(logact_solutes, 0, logact_water)
        return logact

    def gas_activity_function(self, molals_gases, TK, P):
        """
        Activity function for gaseous species

        Parameters
        ----------
        molals_gases: self._np.ndarray
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
        molal_fractions = molals_gases/self._np.sum(molals_gases)
        fugacity_coefficient_term = self._fugacity_coefficient_function(
            molal_fractions, TK, P)
        partial_pressure_term = self._np.log10(P) + self._np.log10(molal_fractions)
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
        Callable[self._np.ndarray, float, float] -> float
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
        res = builder.get_log_equilibrium_constants(self.reactions, TK, PATM)
        if self.backend == 'jax':
            res = jnp.array(res)
        return res
    
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
        res = builder.get_log_equilibrium_constants(self.solid_reactions, TK, PATM)
        if self.backend == 'jax':
            res = jnp.array(res)
        return res

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
        res = builder.get_log_equilibrium_constants(self.gas_reactions, TK, PATM)
        if self.backend == 'jax':
            res = jnp.array(res)
        return res

    def get_diffusion_coefficients(self, TK):
        res = diffusion_coefficients.get_diffusion_coefficients(self.solutes, TK)
        if self.backend == 'jax':
            res = jnp.array(res)
        return res

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
        return self._np.array([constants.ALKALINE_COEFFICIENTS.get(specie, 0.0)
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

    @property
    def _np(self):
        if self.backend == 'numpy':
            return np
        elif self.backend == 'jax':
            return jnp
        
    def _initialize_species_reactions(self,
                                      possible_reactions=None,
                                      possible_solid_reactions=None,
                                      possible_gas_reactions=None):
        return builder.get_species_reaction_from_initial_species(
            self.base_species, possible_reactions,
            possible_solid_reactions, possible_gas_reactions)

    def _make_formula_and_stoich_matrices(self):
        formula_matrix = builder.make_formula_matrix(
            self.species, self.elements)
        stoich_matrix = builder.make_stoich_matrix(
            self.species, self.reactions)
        if self.backend == 'jax':
            formula_matrix = jnp.array(formula_matrix)
            stoich_matrix = jnp.array(stoich_matrix)
        return formula_matrix, stoich_matrix

    def _make_solid_formula_and_stoich_matrices(self):
        if not self.solid_reactions:  # No precipitating solid phases
            solid_formula_matrix = self._np.zeros((self.nelements+1, 0))
            solid_stoich_matrix = self._np.zeros((0, self.nspecies))
        else:
            solid_formula_matrix = builder.make_solid_formula_matrix(
                self.solid_reactions, self.elements)
            solid_stoich_matrix = builder.make_stoich_matrix(
                self.species, self.solid_reactions)
        if self.backend == 'jax':
            solid_formula_matrix = jnp.array(solid_formula_matrix)
            solid_stoich_matrix = jnp.array(solid_stoich_matrix)
        return solid_formula_matrix, solid_stoich_matrix

    def _make_gas_formula_and_stoich_matrices(self):
        if not self.gas_reactions:
            gas_formula_matrix = self._np.zeros((self.nelements+1, 0))
            gas_stoich_matrix = self._np.zeros((0, self.nspecies))
        else:
            gas_formula_matrix = builder.make_gas_formula_matrix(
                self.gas_reactions, self.elements)
            gas_stoich_matrix = builder.make_stoich_matrix(
                self.species, self.gas_reactions)
        if self.backend == 'jax':
            gas_formula_matrix = jnp.array(gas_formula_matrix)
            gas_stoich_matrix = jnp.array(gas_stoich_matrix)
        return gas_formula_matrix, gas_stoich_matrix

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
