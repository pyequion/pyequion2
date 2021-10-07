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
    # S@loga - logKs = 0
    # A_1@x_* - b_1 = 0
    # A_2@logx_* - b_2 = 0
    # x_{*,i} = x_{extended,i} if not m_{1,i} else a
    # With x_{extended} = hstack[1,x]

    logacts = activity_function(molals, TK)
    acts = np.nan_to_num(10**logacts)
    molal_star = np.append(1.0, molals)
    logmolal_star = np.log(molal_star)
    res1mm = stoich_matrix@logacts
    res1 = log_equilibrium_constants - res1mm
    mask_log_ = mask_log if type(
        mask_log) is not np.ndarray else mask_log[..., None]
    mask_ = mask if type(mask) is not np.ndarray else mask[..., None]
    res2mm = balance_matrix_log*(1-mask_log_)@logmolal_star + \
        balance_matrix_log*(mask_log_)@logacts
    res2 = balance_vector_log - res2mm
    res3mm = balance_matrix*(1-mask_)@molal_star + \
        balance_matrix*(mask_)@acts
    res3 = balance_vector - res3mm
    res = np.hstack([res1, res2, res3])

    # Jacobian approximation
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_balance_matrix_log = balance_matrix_log[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    jacobian1 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian2 = -reduced_balance_matrix_log*activity_hessian_diag
    jacobian3 = -reduced_balance_matrix
    jacobian = np.vstack([jacobian1, jacobian2, jacobian3])
    return res, jacobian


def residual_and_jacobian_xlma(molals, molals_p, stability_indexes_p,
                               TK, activity_function,
                               balance_vector,
                               log_equilibrium_constants, log_solubility_constants,
                               balance_matrix, balance_matrix_p,
                               stoich_matrix, stoich_matrix_p):
    # S@loga - logKs = 0
    # A_1@x_* - b_1 = 0
    # A_2@logx_* - b_2 = 0
    # x_{*,i} = x_{extended,i} if not m_{1,i} else a
    # With x_{extended} = hstack[1,x]
    # Here, solid reactions is : reagents positives, solid negatives
    # stability_index_p = -log(saturation) (for solids)
    logacts = activity_function(molals, TK)
    molal_star = np.append(1.0, molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = balance_vector - balance_matrix@molal_star - balance_matrix_p@molals_p
    res4 = log_solubility_constants - stability_indexes_p - \
        stoich_matrix_p@logacts
    res5 = molals_p*stability_indexes_p
    res = np.hstack([res1, res2, res4, res5])

    # Jacobian approximation
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_p = stoich_matrix_p[:, 1:]
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    jacobian11 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian21 = -reduced_balance_matrix
    jacobian41 = -reduced_stoich_matrix_p*activity_hessian_diag
    jacobian51 = np.zeros((molals_p.size, molals.size))
    jacobian12 = np.zeros((jacobian11.shape[0], molals_p.size))
    jacobian22 = -balance_matrix_p
    jacobian42 = np.zeros((jacobian41.shape[0], molals_p.size))
    jacobian52 = np.diag(stability_indexes_p)
    jacobian13 = np.zeros_like(jacobian12)
    jacobian23 = np.zeros_like(jacobian22)
    jacobian43 = -np.identity(stability_indexes_p.size)
    jacobian53 = np.diag(molals_p)
    jacobian = np.block([[jacobian11, jacobian12, jacobian13],
                         [jacobian21, jacobian22, jacobian23],
                         [jacobian41, jacobian42, jacobian43],
                         [jacobian51, jacobian52, jacobian53]])
    return res, jacobian


def residual_and_jacobian_interface_dr(molals, TK, molals_bulk,
                                       activity_function,
                                       log_equilibrium_constants, log_solubility_constants,
                                       balance_matrix, stoich_matrix, stoich_matrix_sol,
                                       transport_constants,
                                       reaction_function, reaction_function_derivative):
    
    #Here, solid reactions is : reagents positives, solid negatives
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_sol = stoich_matrix_sol[:, 1:]
    logacts = activity_function(molals, TK)
    logsatur = stoich_matrix_sol@logacts - log_solubility_constants
    reaction_vector = reaction_function(logsatur, log_solubility_constants)
    transport_vector = transport_constants*(molals_bulk - molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = reduced_balance_matrix@(transport_vector - \
                                   reduced_stoich_matrix_sol.transpose()@reaction_vector)
    res = np.hstack([res1, res2])
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    reaction_vector_prime = reaction_function_derivative(logsatur, log_solubility_constants)
    jacobian1 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian2a = -reduced_balance_matrix*transport_constants
    jacobian2b_ = (reduced_stoich_matrix_sol.transpose()*reaction_vector_prime)@\
                  (reduced_stoich_matrix_sol*activity_hessian_diag)
    jacobian2b = -reduced_balance_matrix@jacobian2b_
    jacobian2 = jacobian2a + jacobian2b
    jacobian = np.vstack([jacobian1, jacobian2])
    return res, jacobian


def residual_and_jacobian_interface_d(molals, TK, molals_bulk,
                                      activity_function,
                                      log_equilibrium_constants,
                                      log_solubility_constants,
                                      balance_matrix,
                                      stoich_matrix,
                                      stoich_matrix_sol,
                                      kernel_matrix_sol,
                                      transport_constants):
    
    #Here, solid reactions is : reagents positives, solid negatives
    reduced_balance_matrix = balance_matrix[:, 1:]
    kernel_reduced_balance_matrix = kernel_matrix_sol@reduced_balance_matrix
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_sol = stoich_matrix_sol[:, 1:]
    logacts = activity_function(molals, TK)
    transport_vector = transport_constants*(molals_bulk - molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = kernel_reduced_balance_matrix@transport_vector
    res3 = log_solubility_constants - stoich_matrix_sol@logacts
    res = np.hstack([res1, res2, res3])
    
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    jacobian1 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian2 = -kernel_reduced_balance_matrix*transport_constants
    jacobian3 = -reduced_stoich_matrix_sol*activity_hessian_diag
    jacobian = np.vstack([jacobian1, jacobian2, jacobian3])
    return res, jacobian


def residual_and_jacobian_interface_mixed(molals, TK, molals_bulk,
                                          activity_function,
                                          log_equilibrium_constants,
                                          log_solubility_constants_1,
                                          log_solubility_constants_2,
                                          balance_matrix,
                                          stoich_matrix,
                                          stoich_matrix_sol_1,
                                          stoich_matrix_sol_2,
                                          kernel_matrix_sol_2,
                                          transport_constants,
                                          reaction_function_1,
                                          reaction_function_derivative_1):
    
    #Here, solid reactions is : reagents positives, solid negatives
    reduced_balance_matrix = balance_matrix[:, 1:]
    kernel_reduced_balance_matrix = kernel_matrix_sol_2@reduced_balance_matrix
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_sol_1 = stoich_matrix_sol_1[:, 1:]
    reduced_stoich_matrix_sol_2 = stoich_matrix_sol_2[:, 1:]
    logacts = activity_function(molals, TK)
    logsatur1 = stoich_matrix_sol_1@logacts - log_solubility_constants_1
    reaction_vector_1 = reaction_function_1(logsatur1, log_solubility_constants_1)
    transport_vector = transport_constants*(molals_bulk - molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = kernel_reduced_balance_matrix@(transport_vector - \
                            kernel_reduced_balance_matrix.transpose()@reaction_vector_1)
    res3 = log_solubility_constants_2 - stoich_matrix_sol_2@logacts
    res = np.hstack([res1, res2, res3])
    
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    reaction_vector_prime_1 = reaction_function_derivative_1(logsatur1,
                                                           log_solubility_constants_1)
    jacobian1 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian2a = -kernel_reduced_balance_matrix*transport_constants
    jacobian2b_ = (reduced_stoich_matrix_sol_1.transpose()*reaction_vector_prime_1)@\
                  (reduced_stoich_matrix_sol_1*activity_hessian_diag)
    jacobian2b = -kernel_reduced_balance_matrix@jacobian2b_
    jacobian2 = jacobian2a + jacobian2b
    jacobian3 = -reduced_stoich_matrix_sol_2*activity_hessian_diag
    jacobian = np.vstack([jacobian1, jacobian2, jacobian3])
    return res, jacobian