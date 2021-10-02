# -*- coding: utf-8 -*-
import numpy as np

from . import builder
from . import converters


class SolutionResult():
    """
    Class for solution of equilibria

    Parameters
    ----------
    TK: float
        Temperature in Kelvin
    base_species: List[str]
        Base aqueous species in system
    elements: List[str]
        Elements in system
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
    """

    def __init__(self, equilibrium_system, x, TK, molals_p=None, solid_phases_p=None):
        self.TK = TK
        self._x_molal = x
        self._x_logact = equilibrium_system.activity_function(x, TK)
        self._x_act = np.nan_to_num(10**self._x_logact)
        self._molals_p = molals_p
        self._solid_phases_p = solid_phases_p
        self.base_species = equilibrium_system.base_species
        self.species = equilibrium_system.species
        self.reactions = equilibrium_system.reactions
        self.solid_reactions = equilibrium_system.solid_reactions
        self.elements = equilibrium_system.elements
        self.formula_matrix = equilibrium_system.formula_matrix
        self.stoich_matrix = equilibrium_system.stoich_matrix
        self.solid_formula_matrix = equilibrium_system.solid_formula_matrix
        self.solid_stoich_matrix = equilibrium_system.solid_stoich_matrix
        self._logsatur = \
            self._build_saturation_indexes()

    @property
    def molals(self):
        """molals"""
        return {self.solutes[i]: self._x_molal[i]
                for i in range(len(self._x_molal))}

    @property
    def concentrations(self):  # mM or mol/m^3
        """Equilibrium concentrations"""
        return {self.solutes[i]: converters.molal_to_mmolar(self._x_molal[i])
                for i in range(len(self._x_molal))}

    @property
    def activities(self):
        """Equilibrium activities"""
        return {self.species[i]: self._x_act[i]
                for i in range(len(self._x_act))}

    @property
    def saturation_indexes(self):
        """Saturation indexes for solids"""
        return {self.phase_names[i]: self._logsatur[i]
                for i in range(len(self._logsatur))}

    @property
    def ionic_strength(self):
        """Ionic strength of system"""
        return 0.5*np.sum(
            self._charge_vector[1:]**2*self._x_molal)

    @property
    def solute_elements(self):  # Ignore H and O
        """Elements ignoring H and O"""
        return self.elements[2:]

    @property
    def solutes(self):  # Ignore H2O
        """Solutes"""
        return self.species[1:]

    @property
    def phase_names(self):
        """Names of solid phases"""
        return [sol_reac['phase_name'] for sol_reac in self.solid_reactions]

    @property
    def solid_molals(self):
        """Solid molals"""
        if self._solid_phases_p is None:
            solid_molals_ = dict()
        else:
            solid_molals_ = dict(zip(self._solid_phases_p, self._molals_p))
        solid_molals = {k: solid_molals_.get(k, 0.0) for k in self.phase_names}
        return solid_molals

    @property
    def elements_molals(self):
        """Molals for elements"""
        balance_vector = self._balance_vector
        return {k: balance_vector[i] for i, k in enumerate(self.solute_elements)}

    @property
    def charge_density(self):
        """Charge density (e/kg)"""
        return np.sum(self.formula_matrix[-1, 1:]*self._x_molal)

    @property
    def ph(self):
        """pH"""
        return -np.log10(self.activities['H+'])

    @property
    def electrical_conductivity(self):
        """Electrical conductivity in S/m"""
        #https://www.aqion.de/site/electrical-conductivity
        ec = 6.2*self.ionic_strength #muS/cm to S/m
        return ec
    
    def _build_saturation_indexes(self):
        logacts = np.log10(self._x_act)
        solid_reactions = self.solid_reactions
        TK = self.TK
        solid_stoich_matrix = self.solid_stoich_matrix
        logiap = solid_stoich_matrix@logacts
        logks = builder.get_log_equilibrium_constants(solid_reactions, TK)
        logsatur = logiap - logks
        return logsatur

    @property
    def _charge_vector(self):
        return self.formula_matrix[-1, :]

    @property
    def _balance_vector(self):
        return self.formula_matrix[2:-1, 1:]@self._x_molal
