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
                              tol=1e-6):
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
    x, res = solver_function(f, x_guess, tol=tol)
    return x, res


def solve_equilibrium_xlma(x_guess, x_guess_p, stability_guess_p,
                           TK, activity_function,
                           balance_vector,
                           log_equilibrium_constants, log_solubility_constants,
                           balance_matrix, balance_matrix_p,
                           stoich_matrix, stoich_matrix_p,
                           solver_function=None,
                           tol=1e-6):
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
    x, res = solver_function(f, x_guess_total, tol=tol)
    molals, molals_p, stability_indexes_p = np.split(x, [ns1, ns2, ns3])[:-1]
    return molals, molals_p, stability_indexes_p, res
