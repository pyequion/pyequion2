# -*- coding: utf-8 -*-
import numpy as np

from .. import builder
from .. import activity
from .. import constants
from .. import eqsolver
from .. import solution
from .. import equilibrium_system
from .. import water_properties
from . import interface_functions
from . import diffusion_coefficients


class InterfaceSystem(equilibrium_system.EquilibriumSystem):
    """
    Extension of EquilibriumSystem for allowing equilibrium calculations 
    at reaction interfaces
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface_phases = None
        self.interface_indexes = None
        self.ninterface = None
        self._flist_reac = []
        self._dflist_reac = []

    def solve_interface_equilibrium_dr(self, TK, molals_bulk,
                                       transport_params,
                                       tol=1e-12, initial_guess='default'):
        """
        TK: float
        xbulk: numpy.ndarray
        tranpost_params: Dict[str]
        """
        assert self.interface_phases is not None
        assert self._flist_reac
        transport_vector = self._get_transport_vector(transport_params, TK)
        activity_function = self.activity_function
        balance_matrix = self.reduced_formula_matrix
        stoich_matrix = self.stoich_matrix
        stoich_matrix_sol = self.solid_stoich_matrix[self.interface_indexes, :]
        reaction_function = self.reaction_function
        reaction_function_derivative = self.reaction_function_derivative
        log_equilibrium_constants = \
            self.get_log_equilibrium_constants(TK)
        log_solubility_constants = self.get_solid_log_equilibrium_constants(TK)
        log_solubility_constants = log_solubility_constants[self.interface_indexes]
        molals_bulk_ = np.array([molals_bulk[k] for k in self.solutes])
        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
        elif isinstance(initial_guess, float):
            x_guess = np.ones(self.nsolutes)*initial_guess
        else:
            x_guess = np.array(initial_guess)
        x, res = eqsolver.solve_equilibrium_interface_dr(x_guess,
                                                              TK,
                                                              molals_bulk_,
                                                              activity_function,
                                                              log_equilibrium_constants,
                                                              log_solubility_constants,
                                                              balance_matrix,
                                                              stoich_matrix,
                                                              stoich_matrix_sol,
                                                              transport_vector,
                                                              reaction_function,
                                                              reaction_function_derivative,
                                                              tol=tol)
        return solution.SolutionResult(self, x, TK), res

    def set_interface_phases(self, phases=None, TK=None):
        if phases is None:
            if TK is None:
                print("Getting all stable phases at 298.15 K")
                TK = 298.15
            else:
                print("Getting all stable phases at %.2f K" % TK)
            phases = builder.get_most_stable_phases(
                self.solid_reactions, TK)
        indexes = self._get_solid_indexes(phases)
        self.interface_phases = phases
        self.interface_indexes = indexes
        self.ninterface = len(phases)

    def set_reaction_functions(self, flist, arglist, dflist=None):
        """
        flist: List[(Callable[[float, float, *args], float] or str)] 
        arglist: List[List]
        dflist: List[(Callable[[float, float, *args], float]] or None)] or None
        """
        # assert sizes are ok
        assert(len(flist) == len(arglist))
        assert(len(flist) == self.ninterface)
        if dflist is None:
            assert all(isinstance(f, str) for f in flist)
            dflist = [None for _ in range(self.ninterface)]
        else:
            assert(len(dflist) == len(flist))
        self._flist_reac = []
        self._dflist_reac = []
        for (f, df, args) in zip(flist, dflist, arglist):
            if isinstance(f, str):
                assert f in interface_functions.INTERFACE_MAP.keys()
                f, df = interface_functions.INTERFACE_MAP[f]
            def f_(logsatur, logksp):
                return f(logsatur, logksp, *args)
            def df_(logsatur, logksp):
                return df(logsatur, logksp, *args)
            self._flist_reac.append(f_)
            self._dflist_reac.append(df_)

    def reaction_function(self, logsatur, logksp):
        res = np.stack([f(logsatur[i], logksp[i])
                        for i, f in enumerate(self._flist_reac)])
        return res

    def reaction_function_derivative(self, logsatur, logksp):
        res = np.stack([df(logsatur[i], logksp[i])
                        for i, df in enumerate(self._dflist_reac)])
        return res

    def _get_transport_vector(self, transport_params, TK):
        """
        transport_params: Dict[str]
        """
        transport_type = transport_params['type']
        if transport_type == 'array':
            transport_vector = transport_params['array']
        elif transport_type == 'dict':
            default_value = transport_params.get('default', np.nan)
            transport_dict = transport_params['dict']
            transport_vector = np.array([transport_dict.get(k, default_value)
                                         for k in self.solutes])
            if default_value is np.nan:
                transport_median = np.nanmedian(transport_vector)
                transport_vector = np.nan_to_num(transport_vector,
                                                 nan=transport_median)
        else:
            # We will use diffusion coefficients everywhere,
            # as well as water density (molality to molarity (in SI))
            rho_water = water_properties.water_density(TK)
            diffusion_coefs_ = {k: diffusion_coefficients.COEFFICIENTS[k]
                                for k in self.solutes
                                if k in diffusion_coefficients.COEFFICIENTS.keys()}
            diffusion_coefs = diffusion_coefficients.diffusion_temp(
                diffusion_coefs_, TK)
            diffusion_median = np.median(list(diffusion_coefs.values()))
            diffusion_vector = np.array([diffusion_coefs.get(k, diffusion_median)
                                         for k in self.solutes])
            if transport_type == 'sphere':
                radius = transport_params['radius']
                transport_vector = rho_water*diffusion_vector/radius
            elif transport_type == 'pipe':
                shear_velocity = transport_params['shear_velocity']
                bturb = transport_params.get(
                    'bturb', constants.TURBULENT_VISCOSITY_CONSTANT)
                scturb = transport_params.get('scturb', 1.0)
                kinematic_viscosity = water_properties.water_kinematic_viscosity(
                    TK)
                transport_vector_a = 3*np.sqrt(3)/2*(bturb/scturb)**(1.0/3)
                transport_vector_b = (
                    diffusion_vector/kinematic_viscosity)**(2.0/3)
                transport_vector_c = shear_velocity*rho_water
                transport_vector = transport_vector_a*transport_vector_b*transport_vector_c
            else:
                raise ValueError("Not valid transport_type")
        return transport_vector
