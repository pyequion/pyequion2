# -*- coding: utf-8 -*-
import functools

import numpy as np

from . import residual_functions
from . import solvers


def solve_equilibrium_solutes(x_guess,
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
                              solver_function=None,
                              tol=1e-6, maxiter=1000):
    """
    Parameters
    ----------
    x_guess : numpy.ndarray
        Initial molals guess
    TK : float
        Temperature of system in Kelvins
    activity_function : Callable
        Function from molals to activities of solute and water, (n,) to (n+1,)
    balance_vector : numpy.ndarray
        Vector of unwarped equilibria
    balance_vector_log : numpy.ndarray
        Vector of warped equilibria
    log_equilibrium_constants : numpy.ndarray
        Vector of log-equilibrium constants
    balance_matrix : numpy.ndarray
        Matrix of unwarped balance equilibria
    balance_matrix_log : numpy.ndarray
        Matrix of warped balance equilibria
    stoich_matrix : numpy.ndarray
        Stoichiometric matrix
    mask : int or numpy.ndarray
        Mask for unwarped equation
    mask_log : int or numpy.ndarray
        Mask for warped equation
    solver_function : None or callable
        If is not None, solver function of f(x) = 0, x_i > 0,
        with access to residual and jacobian.
        If is None, uses constrained newton method for this with
        default parameters
    tol : Solver tolerance
        Tolerance for solver

    returns:
        molal values that solves equilibria, and residual
    """
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    f = functools.partial(residual_functions.residual_and_jacobian_solutes,
                          TK=TK,
                          activity_function=activity_function,
                          balance_vector=balance_vector,
                          balance_vector_log=balance_vector_log,
                          log_equilibrium_constants=log_equilibrium_constants,
                          balance_matrix=balance_matrix,
                          balance_matrix_log=balance_matrix_log,
                          stoich_matrix=stoich_matrix,
                          mask=mask,
                          mask_log=mask_log)
    x, res = solver_function(f, x_guess, tol=tol, maxiter=maxiter)
    return x, res


def solve_equilibrium_xlma(x_guess, x_guess_p, stability_guess_p,
                           TK, activity_function,
                           balance_vector,
                           log_equilibrium_constants, log_solubility_constants,
                           balance_matrix, balance_matrix_p,
                           stoich_matrix, stoich_matrix_p,
                           solver_function=None,
                           tol=1e-6, maxiter=1000):
    """
    Parameters
    ----------
    x_guess : numpy.ndarray
        Initial molals guess
    x_guess_p : numpy.ndarray
        Initial molals precipitates guess
    stability_guess_p : numpy.ndarray
        Initial stability_indexes guess
    TK : float
        Temperature of system in Kelvins
    activity_function : Callable
        Function from molals to activities of solute and water, (n,) to (n+1,)
    balance_vector : numpy.ndarray
        Vector of unwarped equilibria
    log_equilibrium_constants : numpy.ndarray
        Vector of log-equilibrium constants
    log_solubility_constants : numpy.ndarray
        Vector of log-solubility constants
    balance_matrix : numpy.ndarray
        Matrix of unwarped balance equilibria
    balance_matrix_p : numpy.ndarray
        Matrix of unwarped balance equilibria for precipitates
    stoich_matrix : numpy.ndarray
        Stoichiometric matrix
    stoich_matrix_p : numpy.ndarray
        Stoichiometric matrix for solid reaction
    solver_function : None or callable
        If is not None, solver function of f(x) = 0, x_i > 0,
        with access to residual and jacobian.
        If is None, uses constrained newton method for this with
        default parameters
    tol : Solver tolerance
        Tolerance for solver

    returns:
        molal values that solves equilibria, and residual
    """
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    ns1 = x_guess.size
    ns2 = ns1 + x_guess_p.size
    ns3 = ns2 + stability_guess_p.size
    x_guess_total = np.hstack([x_guess, x_guess_p, stability_guess_p])

    def f(x):
        molals, molals_p, stability_indexes_p = np.split(x, [ns1, ns2, ns3])[
            :-1]
        return residual_functions.residual_and_jacobian_xlma(
            molals, molals_p, stability_indexes_p,
            TK, activity_function,
            balance_vector,
            log_equilibrium_constants, log_solubility_constants,
            balance_matrix, balance_matrix_p,
            stoich_matrix, stoich_matrix_p)
    x, res = solver_function(f, x_guess_total, tol=tol, maxiter=maxiter)
    molals, molals_p, stability_indexes_p = np.split(x, [ns1, ns2, ns3])[:-1]
    return molals, molals_p, stability_indexes_p, res

def solve_equilibrium_xlma_2(molals_guess, molals_solids_guess, molals_gases_guess, 
                             stability_indexes_solids_guess, stability_indexes_gases_guess,
                             TK, P, activity_function, activity_function_gas,
                             balance_vector,
                             log_equilibrium_constants, log_solubility_constants, log_gases_constants,
                             balance_matrix, balance_matrix_solids, balance_matrix_gases,
                             stoich_matrix, stoich_matrix_solids, stoich_matrix_gases,
                             solver_function=None, tol=1e-6, maxiter=1000):
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    ns1 = molals_guess.size
    ns2 = ns1 + molals_solids_guess.size
    ns3 = ns2 + molals_gases_guess.size
    ns4 = ns3 + stability_indexes_solids_guess.size
    ns5 = ns4 + stability_indexes_gases_guess.size
    
    x_guess_total = np.hstack([molals_guess, molals_solids_guess, molals_gases_guess, 
                               stability_indexes_solids_guess, stability_indexes_gases_guess])

    def f(x):
        molals, molals_solids, molals_gases, \
        stability_indexes_solids, stability_indexes_gases = \
            np.split(x, [ns1, ns2, ns3, ns4, ns5])[
            :-1]
        return residual_functions.residual_and_jacobian_xlma_2(
            molals, molals_solids, molals_gases, 
            stability_indexes_solids, stability_indexes_gases,
            TK, P, activity_function, activity_function_gas,
            balance_vector,
            log_equilibrium_constants, log_solubility_constants, log_gases_constants,
            balance_matrix, balance_matrix_solids, balance_matrix_gases,
            stoich_matrix, stoich_matrix_solids, stoich_matrix_gases)
    x, res = solver_function(f, x_guess_total, tol=tol, maxiter=maxiter)
    molals, molals_solids, molals_gases, \
    stability_indexes_solids, stability_indexes_gases = \
        np.split(x, [ns1, ns2, ns3, ns4, ns5])[
        :-1]
    return molals, molals_solids, molals_gases, stability_indexes_solids, stability_indexes_gases, res

    
def solve_equilibrium_interface_mixed(x_guess,
                                      TK, molals_bulk,
                                      activity_function,
                                      log_equilibrium_constants,
                                      log_solubility_constants_exp,
                                      log_solubility_constants_imp,
                                      balance_matrix,
                                      stoich_matrix,
                                      stoich_matrix_sol_exp,
                                      stoich_matrix_sol_imp,
                                      kernel_matrix_sol_imp,
                                      transport_constants,
                                      reaction_function_exp,
                                      reaction_function_derivative_exp,
                                      solver_function=None,
                                      tol=1e-6, maxiter=1000):
    """
    Parameters
    ----------
    x_guess : numpy.ndarray
        Initial molals guess
    TK : float
        Temperature of system in Kelvins
    molals_bulk : numpy.ndarray
        Values of molals in the bulk
    activity_function : Callable[numpy.ndarray, numpy.ndarray]
        Function from molals to activities of solute and water, (n,) to (n+1,)
    log_equilibrium_constants : numpy.ndarray
        Vector of log-equilibrium constants
    log_solubility_constants_exp : numpy.ndarray
        Vector of log-solubility constants for explicit interface reactions
    log_solubility_constants_imp : numpy.ndarray
        Vector of log-solubility constants for implicit interface reactions
    balance_matrix : numpy.ndarray
        Matrix of balance equilibria
    stoich_matrix : numpy.ndarray
        Stoichiometric matrix
    stoich_matrix_sol_exp : numpy.ndarray
        Stoichiometric matrix of explicit interface reactions
    stoich_matrix_sol_imp : numpy.ndarray
        Stoichiometric matrix of implicit interface reactions
    kernel_matrix_sol_imp : numpy.ndarray
        Matrix whose transpose has columns composed of a base of 
        ker(stoich_matrix_sol_imp@balance_matrix.T)
    transport_constants : numpy.ndarray
        Transport constants for transport to interface
    reaction_function_exp : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
        Reaction function for explicit interface reactions,
        taking log_saturation and log_solubility_constants
    reaction_function_derivative_exp : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
        Derivative for reaction_function_exp in log-solubility argument
    solver_function : None or callable
        If is not None, solver function of f(x) = 0, x_i > 0,
        with access to residual and jacobian.
        If is None, uses constrained newton method for this with
        default parameters
    tol : Solver tolerance
        Tolerance for solver

    returns:
        molal values that solves equilibria, and residual
    """
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    f = functools.partial(residual_functions.residual_and_jacobian_interface_mixed,
                          TK=TK,
                          molals_bulk=molals_bulk,
                          activity_function=activity_function,
                          log_equilibrium_constants=log_equilibrium_constants,
                          log_solubility_constants_exp=log_solubility_constants_exp,
                          log_solubility_constants_imp=log_solubility_constants_imp,
                          balance_matrix=balance_matrix,
                          stoich_matrix=stoich_matrix,
                          stoich_matrix_sol_exp=stoich_matrix_sol_exp,
                          stoich_matrix_sol_imp=stoich_matrix_sol_imp,
                          kernel_matrix_sol_imp=kernel_matrix_sol_imp,
                          transport_constants=transport_constants,
                          reaction_function_exp=reaction_function_exp,
                          reaction_function_derivative_exp=reaction_function_derivative_exp)
    x, res = solver_function(f, x_guess, tol=tol, maxiter=maxiter)
    return x, res


def solve_equilibrium_interface_slack_a(x_guess, reaction_imp_guess, stability_imp_guess,
                                        TK, molals_bulk,
                                        activity_function,
                                        log_equilibrium_constants,
                                        log_solubility_constants_exp,
                                        log_solubility_constants_imp,
                                        balance_matrix,
                                        stoich_matrix,
                                        stoich_matrix_sol_exp,
                                        stoich_matrix_sol_imp,
                                        transport_constants,
                                        reaction_function_exp,
                                        reaction_function_derivative_exp,
                                        solver_function=None,
                                        tol=1e-6, maxiter=1000):
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    ns1 = x_guess.size
    ns2 = ns1 + reaction_imp_guess.size
    ns3 = ns2 + stability_imp_guess.size
    x_guess_total = np.hstack([x_guess, reaction_imp_guess, stability_imp_guess])
    def f(x):
        x_guess, reaction_imp_guess, stability_imp_guess = np.split(x, [ns1, ns2, ns3])[:-1]
        return residual_functions.residual_and_jacobian_interface_slack_a(
                                      x_guess,
                                      reaction_imp_guess,
                                      stability_imp_guess,
                                      TK, molals_bulk,
                                      activity_function,
                                      log_equilibrium_constants,
                                      log_solubility_constants_exp,
                                      log_solubility_constants_imp,
                                      balance_matrix,
                                      stoich_matrix,
                                      stoich_matrix_sol_exp,
                                      stoich_matrix_sol_imp,
                                      transport_constants,
                                      reaction_function_exp,
                                      reaction_function_derivative_exp)
    x, res = solver_function(f, x_guess_total, tol=tol, maxiter=maxiter)
    molals, reaction_imp_guess, stability_imp_guess = np.split(x, [ns1, ns2, ns3])[:-1]
    return molals, reaction_imp_guess, stability_imp_guess, res


def solve_equilibrium_interface_slack_b(x_guess, reaction_imp_guess, stability_imp_guess,
                                        TK, molals_bulk,
                                        activity_function,
                                        log_equilibrium_constants,
                                        log_solubility_constants_exp,
                                        log_solubility_constants_imp,
                                        balance_matrix,
                                        stoich_matrix,
                                        stoich_matrix_sol_exp,
                                        stoich_matrix_sol_imp,
                                        transport_constant,
                                        relative_diffusion_vectors,
                                        reaction_function_exp,
                                        reaction_function_derivative_exp,
                                        solver_function=None,
                                        tol=1e-6, maxiter=1000):
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    ns1 = x_guess.size
    ns2 = ns1 + reaction_imp_guess.size
    ns3 = ns2 + stability_imp_guess.size
    x_guess_total = np.hstack([x_guess, reaction_imp_guess, stability_imp_guess])
    def f(x):
        x_guess, reaction_imp_guess, stability_imp_guess = np.split(x, [ns1, ns2, ns3])[:-1]
        return residual_functions.residual_and_jacobian_interface_slack_b(
                                      x_guess,
                                      reaction_imp_guess,
                                      stability_imp_guess,
                                      TK, molals_bulk,
                                      activity_function,
                                      log_equilibrium_constants,
                                      log_solubility_constants_exp,
                                      log_solubility_constants_imp,
                                      balance_matrix,
                                      stoich_matrix,
                                      stoich_matrix_sol_exp,
                                      stoich_matrix_sol_imp,
                                      transport_constant,
                                      relative_diffusion_vectors,
                                      reaction_function_exp,
                                      reaction_function_derivative_exp)
    x, res = solver_function(f, x_guess_total, tol=tol, maxiter=maxiter)
    molals, reaction_imp_guess, stability_imp_guess = np.split(x, [ns1, ns2, ns3])[:-1]
    return molals, reaction_imp_guess, stability_imp_guess, res
