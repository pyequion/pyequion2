# -*- coding: utf-8 -*-
import numpy as np


def residual_and_jacobian_solutes(molals, TK, activity_function,
                                  balance_vector, balance_vector_log,
                                  log_equilibrium_constants,
                                  balance_matrix, balance_matrix_log,
                                  stoich_matrix, unwarped_activity_mask, 
                                  log_activity_mask):
    """
    Parameters
    ----------
    molals : numpy.ndarray
        Value of molals of solute
    TK : float
        Temperature of system in Kelvins
    activity_function : Callable
        Function from molals to activities of solute and water, (n,) to (n+1,)
    balance_vector : numpy.ndarray
        Vector of molal (unwarped) equilibria
    balance_vector_log : numpy.ndarray
        Vector of logact (warped) equilibria
    log_equilibrium_constants : numpy.ndarray
        Vector of log-equilibrium constants
    balance_matrix_log: numpy.ndarray
        Matrix of molal balance equilibria
    balance_matrix
    """
#    reduced_balance_matrix_molal = balance_matrix_molal[:, 1:]
#    reduced_balance_matrix_logact = balance_matrix_logact[:, 1:]
#    reduced_stoich_matrix = stoich_matrix[:,1:]
    logacts = activity_function(molals,TK)
    extended_molals = np.append(1.0, molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = balance_vector_logact - balance_matrix_logact@logacts
    res3 = balance_vector_molal - reduced_balance_matrix_molal@molals
    res = np.hstack([res1, res2, res3])
    
    #Jacobian approximation
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    jacobian1 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian2 = -reduced_balance_matrix_logact*activity_hessian_diag
    jacobian3 = -reduced_balance_matrix_molal
    jacobian = np.vstack([jacobian1, jacobian2, jacobian3])
    return res,jacobian


#def residual_and_jacobian_solutes(molals,TK,activity_function,
#                                  formula_vector,log_equilibrium_constants,
#                                  stoich_matrix,formula_matrix):
#    reduced_formula_matrix = formula_matrix[:,1:]
#    reduced_stoich_matrix = stoich_matrix[:,1:]
#    logacts = activity_function(molals,TK)
#    res1 = log_equilibrium_constants - stoich_matrix@logacts
#    res2 = reduced_formula_matrix@molals - formula_vector
#    res = np.hstack([res1,res2])
#    
#    #Jacobian approximation
#    mole_fractions = molals/(np.sum(molals))
#    activity_hessian_diag = (1-mole_fractions)/molals
#    upper_jacobian = -reduced_stoich_matrix*activity_hessian_diag
#    lower_jacobian = reduced_formula_matrix
#    jacobian = np.vstack([upper_jacobian,lower_jacobian])
#    return res,jacobian