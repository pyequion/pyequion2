# -*- coding: utf-8 -*-
import numpy as np

from . import builder
from . import converters


class SolutionResult():
    def __init__(self,equilibrium_system, x, TK, molals_p=None, solid_phases_p=None):
        self.TK = TK
        self._x_molal = x
        self._x_act = 10**equilibrium_system.activity_function(x, TK)
        self._molals_p = molals_p
        self._solid_phases_p = solid_phases_p
        self.base_species = equilibrium_system.base_species
        self.species = equilibrium_system.species
        self.reactions = equilibrium_system.reactions
        self.solid_reactions = equilibrium_system.solid_reactions
        self.elements = equilibrium_system.elements
        self.formula_matrix = equilibrium_system.formula_matrix
        self.stoich_matrix = equilibrium_system.stoich_matrix
        self.solid_molals = \
            self.build_solid_molals()
        self._logsatur = \
            self.build_saturation_indexes()
    
    def build_saturation_indexes(self):
        logacts = np.log10(self._x_act)
        species = self.species
        solid_reactions = self.solid_reactions
        TK = self.TK
        solid_stoich_matrix = builder.make_stoich_matrix(species, solid_reactions)
        logiap = solid_stoich_matrix@logacts
        logks = builder.get_log_equilibrium_constants(solid_reactions, TK)
        logsatur = logiap - logks
        return logsatur
    
    def build_solid_molals(self):
        solid_molals_ = dict(zip(self._solid_phases_p, self._molals_p))
        solid_molals = {k: solid_molals_.get(k, 0.0) for k in self.phase_names}
        return solid_molals
    
    @property
    def equilibrium_molals(self):
        return {self.solutes[i]:self._x_molal[i] 
                for i in range(len(self._x_molal))}
    
    @property
    def equilibrium_concentrations(self): #mM or mol/m^3
        return {self.solutes[i]:converters.molal_to_mmolar(self._x_molal[i]) 
                for i in range(len(self._x_molal))}

    @property
    def equilibrium_activities(self):
        return {self.species[i]:self._x_act[i] 
                for i in range(len(self._x_act))}
    
    @property
    def saturation_indexes(self):
        return {self.phase_names[i]:self._logsatur[i]
                for i in range(len(self._logsatur))}

    @property
    def ionic_strength(self):
        return 0.5*np.sum(
                self.charge_vector[1:]**2*self._x_molal)

    @property
    def solute_elements(self): #Ignore H and O
        return self.elements[2:]

    @property
    def solutes(self): #Ignore H2O
        return self.species[1:]

    @property
    def charge_vector(self):
        return self.formula_matrix[-1,:]
    
    @property
    def phase_names(self):
        return [sol_reac['phase_name'] for sol_reac in self.solid_reactions]