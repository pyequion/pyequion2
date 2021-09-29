# -*- coding: utf-8 -*-
import numpy as np


def residual_and_jacobian_solutes(molals, TK, activity_function,
                                  balance_vector, balance_vector_log,
                                  log_equilibrium_constants,
                                  balance_matrix, balance_matrix_log,
                                  stoich_matrix, mask, 
                                  mask_log):    
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
    mask : numpy.ndarray
        Mask for unwarped equation
    mask_log : numpy.ndarray
        Mask for warped equation
        
    Returns:
        residual and jacobian of equilibrium equation
    """
    #S@loga - logKs = 0
    #A_1@x_* - b_1 = 0
    #A_2@logx_* - b_2 = 0
    #x_{*,i} = x_{extended,i} if not m_{1,i} else a
    #With x_{extended} = hstack[1,x]
    
    logacts = activity_function(molals,TK)
    molal_star = np.append(1.0, molals)*(1-mask) + np.exp(logacts)*mask
    logmolal_star = np.log(molal_star)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = balance_vector_log - balance_matrix_log@logmolal_star
    res3 = balance_vector - balance_matrix@molal_star
    res = np.hstack([res1, res2, res3])
    
    #Jacobian approximation
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_balance_matrix_log = balance_matrix_log[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:,1:]
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    jacobian1 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian2 = -reduced_balance_matrix_log*activity_hessian_diag
    jacobian3 = -reduced_balance_matrix
    jacobian = np.vstack([jacobian1, jacobian2, jacobian3])
    return res, jacobian


def residual_and_jacobian_xlma(molals, molals_p, stability_indexes_p,
                               TK, activity_function,
                               balance_vector, balance_vector_log,
                               log_equilibrium_constants, log_solubility_constants,
                               balance_matrix, balance_matrix_log, balance_matrix_p,
                               stoich_matrix, stoich_matrix_p,
                               mask, mask_log):
    #S@loga - logKs = 0
    #A_1@x_* - b_1 = 0
    #A_2@logx_* - b_2 = 0
    #x_{*,i} = x_{extended,i} if not m_{1,i} else a
    #With x_{extended} = hstack[1,x]
    #Here, solid reactions is : reagents positives, solid negatives
    #stability_index_p = -log(saturation)
    logacts = activity_function(molals,TK)
    molal_star = np.append(1.0, molals)*(1-mask) + np.exp(logacts)*mask
    logmolal_star = np.log(molal_star)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = balance_vector - balance_matrix@molal_star - balance_matrix_p@molals_p
    res3 = balance_vector_log - balance_matrix_log@logmolal_star
    res4 = log_solubility_constants - stability_indexes_p - \
            stoich_matrix_p@logacts
    res5 = molals_p*stability_indexes_p
    res = np.hstack([res1, res2, res3, res4, res5])
    
    #Jacobian approximation
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_balance_matrix_log = balance_matrix_log[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_p = stoich_matrix_p[:, 1:]
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    jacobian11 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian21 = -reduced_balance_matrix
    jacobian31 = -reduced_balance_matrix_log*activity_hessian_diag
    jacobian41 = -reduced_stoich_matrix_p*activity_hessian_diag
    jacobian51 = np.zeros((molals_p.size, molals.size))
    jacobian12 = np.zeros((jacobian11.shape[0], molals_p.size))
    jacobian22 = -balance_matrix_p
    jacobian32 = np.zeros((jacobian31.shape[0], molals_p.size))
    jacobian42 = np.zeros((jacobian41.shape[0], molals_p.size))
    jacobian52 = np.diag(stability_indexes_p)
    jacobian13 = np.zeros_like(jacobian12)
    jacobian23 = np.zeros_like(jacobian22)
    jacobian33 = np.zeros_like(jacobian32)
    jacobian43 = -np.identity(stability_indexes_p.size)
    jacobian53 = np.diag(molals_p)
    jacobian = np.block([[jacobian11, jacobian12, jacobian13],
                         [jacobian21, jacobian22, jacobian23],
                         [jacobian31, jacobian32, jacobian33],
                         [jacobian41, jacobian42, jacobian43],
                         [jacobian51, jacobian52, jacobian53]])
    return res, jacobian