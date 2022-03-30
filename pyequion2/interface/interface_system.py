# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg

from . import interface_functions
from . import diffusion_coefficients
from . import interface_solution
from .. import builder
from .. import constants
from .. import eqsolver
from .. import equilibrium_system
from .. import water_properties


class InterfaceSystem(equilibrium_system.EquilibriumSystem):
    """
    Extension of EquilibriumSystem for allowing equilibrium calculations 
    at reaction interfaces
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface_phases = []
        self.explicit_interface_phases = []
        self.implicit_interface_phases = []
        self.ninterface = 0
        self._interface_indexes = []
        self._explicit_interface_indexes = []
        self._implicit_interface_indexes = []
        self._explicit_flist_reac = []
        self._explicit_dflist_reac = []

    def solve_interface_equilibrium(self, TK, molals_bulk,
                                    transport_params,
                                    PATM=1.0,
                                    tol=1e-12, initial_guess='default',
                                    transport_model="A"):
        """
        TK: float
        xbulk: numpy.ndarray
        tranpost_params: Dict[str]
        """
        assert(self.explicit_interface_phases or self.implicit_interface_phases)
        activity_function = self.activity_function
        balance_matrix = self.reduced_formula_matrix
        stoich_matrix = self.stoich_matrix
        log_equilibrium_constants = self.get_log_equilibrium_constants(TK, PATM)
        stoich_matrix_sol_exp = self.solid_stoich_matrix[self._explicit_interface_indexes, :]
        stoich_matrix_sol_imp = self.solid_stoich_matrix[self._implicit_interface_indexes, :]
        log_solubility_constants = self.get_solid_log_equilibrium_constants(TK, PATM)
        log_solubility_constants_exp = log_solubility_constants[self._explicit_interface_indexes]
        log_solubility_constants_imp = log_solubility_constants[self._implicit_interface_indexes]
        reaction_function_exp = self.explicit_reaction_function
        reaction_function_derivative_exp = self.explicit_reaction_function_derivative
        
        molals_bulk_ = np.array([molals_bulk[k] for k in self.solutes])

        if initial_guess == 'default':
            x_guess = np.ones(self.nsolutes)*0.1
            reaction_imp_guess = np.ones(len(self._implicit_interface_indexes))*0.1
            stability_imp_guess = np.zeros(len(self._implicit_interface_indexes))
        elif initial_guess == 'bulk':
            x_guess = molals_bulk_.copy()
            reaction_imp_guess = np.ones(len(self._implicit_interface_indexes))*0.1
            stability_imp_guess = np.zeros(len(self._implicit_interface_indexes))
        elif isinstance(initial_guess, float):
            x_guess = np.ones(self.nsolutes)*initial_guess
            reaction_imp_guess = np.ones(len(self._implicit_interface_indexes))*initial_guess
            stability_imp_guess = np.zeros(len(self._implicit_interface_indexes))
        else:
            x_guess, reaction_imp_guess, stability_imp_guess = initial_guess
        
        if transport_model == "A":
            transport_constant = self._get_transport_vector_a(transport_params, TK)
            relative_diffusion_vector = None
            x, reaction_imp, stability_imp, res = \
                eqsolver.solve_equilibrium_interface_slack_a(x_guess,
                                                             reaction_imp_guess,
                                                             stability_imp_guess,
                                                             TK, molals_bulk_,
                                                             activity_function,
                                                             log_equilibrium_constants,
                                                             log_solubility_constants_exp,
                                                             log_solubility_constants_imp,
                                                             balance_matrix,
                                                             stoich_matrix,
                                                             stoich_matrix_sol_exp,
                                                             stoich_matrix_sol_imp,
                                                             transport_constant,
                                                             reaction_function_exp,
                                                             reaction_function_derivative_exp)
        elif transport_model == "B":
            transport_constant, relative_diffusion_vector = self._get_transport_vector_b(transport_params, TK)
            x, reaction_imp, stability_imp, res = \
                eqsolver.solve_equilibrium_interface_slack_b(x_guess,
                                                             reaction_imp_guess,
                                                             stability_imp_guess,
                                                             TK, molals_bulk_,
                                                             activity_function,
                                                             log_equilibrium_constants,
                                                             log_solubility_constants_exp,
                                                             log_solubility_constants_imp,
                                                             balance_matrix,
                                                             stoich_matrix,
                                                             stoich_matrix_sol_exp,
                                                             stoich_matrix_sol_imp,
                                                             transport_constant,
                                                             relative_diffusion_vector,
                                                             reaction_function_exp,
                                                             reaction_function_derivative_exp)
        solution = interface_solution.InterfaceSolutionResult(self,
                                                              x,
                                                              TK,
                                                              molals_bulk_,
                                                              transport_constant,
                                                              reaction_imp,
                                                              relative_diffusion_vector)
        stats = dict()
        stats['res'] = res
        stats['x'] = (x, reaction_imp, stability_imp)
        return solution, stats

    def set_interface_phases(self,
                             phases=None,
                             TK=298.15,
                             PATM=1.0,
                             fill_defaults=False):
        if phases is None:
            print("Getting all stable phases at %.2f K and %.2f ATM"%(TK, PATM))
            phases = builder.get_most_stable_phases(
                self.solid_reactions, TK, PATM)
        indexes = self.get_solid_indexes(phases)
        self.interface_phases = phases
        self.interface_indexes = indexes
        self.ninterface = len(phases)
        self.set_reaction_functions(fill_defaults=fill_defaults)

    def set_reaction_functions(self, reaction_dict=None, fill_defaults=False):
        """
        reaction_dict: Dict[str]
            Dictionary where keys are phase names and values is a tuple 
            of format (f, args, df), where
            flist: Callable[[float, float, *args], float] or str
            arglist: List
            dflist: Callable[[float, float, *args], float] or None)] or None
        """
        if reaction_dict is None:
            reaction_dict = dict()
        if fill_defaults:
            for phase, (fname, args) in interface_functions.SPECIFIC_SOLIDS_MODEL.items():
                if phase in self.solid_phase_names:
                    reaction_dict[phase] = (fname, args, None)
        self._split_implicit_explicit(reaction_dict)
        self._explicit_flist_reac = []
        self._explicit_dflist_reac = []
        for phase in self.explicit_interface_phases:
            f, args, df = reaction_dict[phase]
            if isinstance(f, str):
                assert f in interface_functions.INTERFACE_MAP.keys()
                f, df = interface_functions.INTERFACE_MAP[f]
            def f_(logsatur, logksp, TK):
                return f(logsatur, logksp, TK, *args)
            def df_(logsatur, logksp, TK):
                return df(logsatur, logksp, TK, *args)
            self._explicit_flist_reac.append(f_)
            self._explicit_dflist_reac.append(df_)

    def explicit_reaction_function(self, logsatur, logksp, TK):
        if not self._explicit_flist_reac: #Case no explicit reactions
            res = np.zeros((0,))
        else:
            res = np.stack([f(logsatur[i], logksp[i], TK)
                            for i, f in enumerate(self._explicit_flist_reac)])
        return res

    def explicit_reaction_function_derivative(self, logsatur, logksp, TK):
        if not self._explicit_dflist_reac:
            res = np.zeros((0,))
        else:
            res = np.stack([df(logsatur[i], logksp[i], TK)
                            for i, df in enumerate(self._explicit_dflist_reac)])
        return res
    
    def _split_implicit_explicit(self, reaction_dict):
        explicit_interface_phases = []
        implicit_interface_phases = []
        explicit_interface_indexes = []
        implicit_interface_indexes = []
        for phase in self.interface_phases:
            if phase in reaction_dict.keys():
                explicit_interface_indexes.append(self.interface_indexes_dict[phase])
                explicit_interface_phases.append(phase)
            else:
                implicit_interface_indexes.append(self.interface_indexes_dict[phase])
                implicit_interface_phases.append(phase)
        self._explicit_interface_indexes = explicit_interface_indexes
        self._implicit_interface_indexes = implicit_interface_indexes
        self.explicit_interface_phases = explicit_interface_phases
        self.implicit_interface_phases = implicit_interface_phases
        
    def _get_transport_vector_a(self, transport_params, TK):
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

    def _get_transport_vector_b(self, transport_params, TK):
        """
        transport_params: Dict[str]
        """
        transport_type = transport_params['type']
        #In model B, this is necessary everywhere
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
        relative_diffusion_vector = diffusion_vector/diffusion_median
        
        if transport_type == 'value':
            transport_constant = transport_params['value']
        elif transport_type == 'array':
            transport_vector = transport_params['array']
            transport_constant = np.median(transport_vector)
        elif transport_type == 'dict':
            default_value = transport_params.get('default', np.nan)
            transport_dict = transport_params['dict']
            transport_vector = np.array([transport_dict.get(k, default_value)
                                         for k in self.solutes])
            if default_value is np.nan:
                transport_median = np.nanmedian(transport_vector)
                transport_vector = np.nan_to_num(transport_vector,
                                                 nan=transport_median)
            transport_constant = np.median(transport_vector)
        else:
            if transport_type == 'sphere':
                radius = transport_params['radius']
                transport_constant = rho_water*diffusion_median/radius
            elif transport_type == 'pipe':
                shear_velocity = transport_params['shear_velocity']
                bturb = transport_params.get(
                    'bturb', constants.TURBULENT_VISCOSITY_CONSTANT)
                scturb = transport_params.get('scturb', 1.0)
                kinematic_viscosity = water_properties.water_kinematic_viscosity(
                    TK)
                transport_constant_a = 3*np.sqrt(3)/2*(bturb/scturb)**(1.0/3)
                transport_constant_b = (
                    diffusion_median/kinematic_viscosity)**(2.0/3)
                transport_constant_c = shear_velocity*rho_water
                transport_constant = transport_constant_a*transport_constant_b*transport_constant_c
            else:
                raise ValueError("Not valid transport_type")
        return transport_constant, relative_diffusion_vector
    
    @property
    def interface_indexes_dict(self):
        return {k: v for k, v
                in zip(self.interface_phases, self.interface_indexes)}

    @property
    def _explicit_interface_indexes_dict(self):
        return {k: v for k, v
                in zip(self.explicit_interface_phases, self._explicit_interface_indexes)}
        
    @property
    def _implicit_interface_indexes_dict(self):
        return {k: v for k, v
                in zip(self.implicit_interface_phases, self._implicit_interface_indexes)}


#Auxiliary functions
def _get_kernel_matrix_implicit(balance_matrix,stoich_matrix_imp):
    A = stoich_matrix_imp@balance_matrix.transpose()
    if A.shape[0] == 0: #Special case, return identity matrix
        Bt = np.identity(A.shape[1])
    else:
        Bt = scipy.linalg.null_space(A)
    B = Bt.transpose()
    return B
        
        
        