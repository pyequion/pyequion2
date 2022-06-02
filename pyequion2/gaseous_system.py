# -*- coding: utf-8 -*-
import numpy as np

from . import builder
from . import fugacity


class InertGaseousSystem(object):
    def __init__(self, possible_gases, database_files=None,
                 fugacity_model="IDEAL"):
        if database_files is None:
            database_files = builder.DEFAULT_DB_FILES
        possible_gas_reactions =  \
            builder.get_all_possible_gas_reactions(database_files)
        _, self.gaseous_reactions = builder._get_species_reactions_from_compounds(
                set(possible_gases), possible_gas_reactions)
        self.gas_names = [r['phase_name'] for r in self.gaseous_reactions]
        self.gas_indexes = {g: i for i, g in enumerate(self.gas_names)}
        self.set_fugacity_coefficient_function(fugacity_model)
    
    def get_fugacity(self, molals_gases, TK, P):
        molals_gases_array = np.zeros(len(self.gas_names))        
        for key, value in molals_gases.items():
            molals_gases_array[self.gas_indexes[key]] = value
        activity_array = self.fugacity_function(molals_gases_array,
                                                TK,
                                                P)
        activity_map = {phase: activity_array[self.gas_indexes[phase]]
                        for phase in self.gas_names}
        return activity_map
        
    def fugacity_function(self, molals_gases, TK, P):
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
        molal_fractions = molals_gases/np.sum(molals_gases)
        fugacity_coefficient_term = self._fugacity_coefficient_function(
            molal_fractions, TK, P)
        partial_pressure_term = np.log10(P) + np.log10(molal_fractions)
        logact = fugacity_coefficient_term + partial_pressure_term
        return logact
    
    def set_fugacity_coefficient_function(self,
                                          fugacity_model="IDEAL"):
        """
        Returns
        -------
        Callable[np.ndarray, float, float] -> float
            Log-fugacity function, accepting molal_fractions, temperature (TK), and pressure (ATM)
        """
        gaseous_reactions = self.gaseous_reactions
        if fugacity_model == "IDEAL":
            self._fugacity_coefficient_function = lambda x, TK, P: 0.0
        elif fugacity_model == "PENGROBINSON":
            self._fugacity_coefficient_function = \
                fugacity.make_peng_robinson_fugacity_function(gaseous_reactions)
        else:
            raise NotImplementedError