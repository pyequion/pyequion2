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


#COUNTER = 0
def residual_and_jacobian_xlma_2(molals, molals_solids, molals_gases, 
                                 stability_indexes_solids, stability_indexes_gases,
                                 TK, P, activity_function, activity_function_gas,
                                 balance_vector,
                                 log_equilibrium_constants, log_solubility_constants, log_gases_constants,
                                 balance_matrix, balance_matrix_solids, balance_matrix_gases,
                                 stoich_matrix, stoich_matrix_solids, stoich_matrix_gases):
    # S@loga - logKs = 0
    # A_1@x_* - b_1 = 0
    # A_2@logx_* - b_2 = 0
    # x_{*,i} = x_{extended,i} if not m_{1,i} else a
    # With x_{extended} = hstack[1,x]
    # Here, solid reactions is : reagents positives, solid negatives
    # stability_index_p = -log(saturation) (for solids)
    logacts = activity_function(molals, TK)
    logacts_gases = activity_function_gas(molals_gases, TK, P)
    
    molal_star = np.append(1.0, molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = balance_vector - \
           balance_matrix@molal_star - \
           balance_matrix_solids@molals_solids - \
           balance_matrix_gases@molals_gases
    res3 = log_solubility_constants - stability_indexes_solids - \
        stoich_matrix_solids@logacts
    res4 = log_gases_constants - stability_indexes_gases - \
           stoich_matrix_gases@logacts + logacts_gases

    res5 = molals_solids*stability_indexes_solids
    res6 = molals_gases*stability_indexes_gases
    res = np.hstack([res1, res2, res3, res4, res5, res6])

#    print(molals, molals_solids, molals_gases, 
#                                 stability_indexes_solids, stability_indexes_gases)
#    print(res3, res5)
#    global COUNTER
#    COUNTER += 1
#    print(COUNTER)
#    if COUNTER == 70:
#        raise KeyError
    #lines: res1, res2, res3, res4, res5, res6
    #columns: molals, molals_solids, molals_gases, stability_indexes_solids, stability_indexes_gases
    # Jacobian approximation
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_solids = stoich_matrix_solids[:, 1:]
    reduced_stoich_matrix_gases = stoich_matrix_gases[:, 1:]    
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    gas_mole_fractions = molals_gases/(np.sum(molals_gases))
    activity_gas_hessian_diag = (1-gas_mole_fractions)/molals_gases
    
    jacobian11 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian21 = -reduced_balance_matrix
    jacobian31 = -reduced_stoich_matrix_solids*activity_hessian_diag
    jacobian41 = -reduced_stoich_matrix_gases*activity_hessian_diag
    jacobian51 = np.zeros((res5.size, molals.size))
    jacobian61 = np.zeros((res6.size, molals.size))    
    jacobian12 = np.zeros((res1.size, molals_solids.size))
    jacobian22 = -balance_matrix_solids
    jacobian32 = np.zeros((res3.size, molals_solids.size))
    jacobian42 = np.zeros((res4.size, molals_solids.size))
    jacobian52 = np.diag(stability_indexes_solids)
    jacobian62 = np.zeros((res6.size, molals_solids.size))
    jacobian13 = np.zeros((res1.size, molals_gases.size))
    jacobian23 = -balance_matrix_gases
    jacobian33 = np.zeros((res3.size, molals_gases.size))
    jacobian43 = np.diag(activity_gas_hessian_diag)
    jacobian53 = np.zeros((res5.size, molals_gases.size))
    jacobian63 = np.diag(stability_indexes_gases)
    jacobian14 = np.zeros((res1.size, stability_indexes_solids.size))
    jacobian24 = np.zeros((res2.size, stability_indexes_solids.size))
    jacobian34 = -np.identity(stability_indexes_solids.size)
    jacobian44 = np.zeros((res4.size, stability_indexes_solids.size))
    jacobian54 = np.diag(molals_solids)
    jacobian64 = np.zeros((res6.size, stability_indexes_solids.size))
    jacobian15 = np.zeros((res1.size, stability_indexes_gases.size))
    jacobian25 = np.zeros((res2.size, stability_indexes_gases.size))
    jacobian35 = np.zeros((res3.size, stability_indexes_gases.size))
    jacobian45 = -np.identity(stability_indexes_gases.size)
    jacobian55 = np.zeros((res5.size, stability_indexes_gases.size))
    jacobian65 = np.diag(molals_gases)

    jacobian = np.block([[jacobian11, jacobian12, jacobian13, jacobian14, jacobian15],
                         [jacobian21, jacobian22, jacobian23, jacobian24, jacobian25],
                         [jacobian31, jacobian32, jacobian33, jacobian34, jacobian35],
                         [jacobian41, jacobian42, jacobian43, jacobian44, jacobian45],
                         [jacobian51, jacobian52, jacobian53, jacobian54, jacobian55],
                         [jacobian61, jacobian62, jacobian63, jacobian64, jacobian65]])
    return res, jacobian


def residual_and_jacobian_interface_slack_a(molals, reaction_vector_imp,
                                            stability_indexes_imp,
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
                                            reaction_function_derivative_exp):
    
    #Here, solid reactions is : reagents positives, solid negatives
    nn = molals.size
    nri = reaction_vector_imp.size
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_sol_exp = stoich_matrix_sol_exp[:, 1:]
    reduced_stoich_matrix_sol_imp = stoich_matrix_sol_imp[:, 1:]
    logacts = activity_function(molals, TK)
    logsatur_exp = stoich_matrix_sol_exp@logacts - log_solubility_constants_exp
    reaction_vector_exp = reaction_function_exp(logsatur_exp, log_solubility_constants_exp, TK)
    transport_vector = transport_constants*(molals_bulk - molals)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = reduced_balance_matrix@(transport_vector - \
                            reduced_stoich_matrix_sol_exp.transpose()@reaction_vector_exp - \
                            reduced_stoich_matrix_sol_imp.transpose()@reaction_vector_imp)
    res3 = log_solubility_constants_imp - stoich_matrix_sol_imp@logacts - stability_indexes_imp
    res4 = reaction_vector_imp*stability_indexes_imp
    res = np.hstack([res1, res2, res3, res4])
    
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    reaction_vector_prime_exp = reaction_function_derivative_exp(logsatur_exp,
                                                           log_solubility_constants_exp,
                                                           TK)
    
    #rows: res1, res2, res3, res4
    #columns: molals, reaction_vector_imp, satbility_indexes_imp
    jacobian11 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian12 = np.zeros((jacobian11.shape[0], nri))
    jacobian13 = np.zeros((jacobian11.shape[0], nri))
    jacobian21a = -reduced_balance_matrix*transport_constants
    jacobian21b_ = (reduced_stoich_matrix_sol_exp.transpose()*reaction_vector_prime_exp)@\
                  (reduced_stoich_matrix_sol_exp*activity_hessian_diag)
    jacobian21b = -reduced_balance_matrix@jacobian21b_
    jacobian21 = jacobian21a + jacobian21b
    jacobian22 = -reduced_balance_matrix@reduced_stoich_matrix_sol_imp.transpose()
    jacobian23 = np.zeros((jacobian21.shape[0], nri))
    jacobian31 = -reduced_stoich_matrix_sol_imp*activity_hessian_diag
    jacobian32 = np.zeros((nri, nri))
    jacobian33 = -np.eye(nri)
    jacobian41 = np.zeros((nri, nn))
    jacobian42 = np.diag(stability_indexes_imp)
    jacobian43 = np.diag(reaction_vector_imp)
    jacobian = np.block([[jacobian11, jacobian12, jacobian13],
                         [jacobian21, jacobian22, jacobian23],
                         [jacobian31, jacobian32, jacobian33],
                         [jacobian41, jacobian42, jacobian43]])
    return res, jacobian


def residual_and_jacobian_interface_slack_b(molals, reaction_vector_imp,
                                            stability_indexes_imp,
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
                                            relative_diffusion_vector,
                                            reaction_function_exp,
                                            reaction_function_derivative_exp):
    
    #Here, solid reactions is : reagents positives, solid negatives
    nn = molals.size
    nri = reaction_vector_imp.size
    reduced_balance_matrix = balance_matrix[:, 1:]
    reduced_stoich_matrix = stoich_matrix[:, 1:]
    reduced_stoich_matrix_sol_exp = stoich_matrix_sol_exp[:, 1:]
    reduced_stoich_matrix_sol_imp = stoich_matrix_sol_imp[:, 1:]
    logacts = activity_function(molals, TK)
    logsatur_exp = stoich_matrix_sol_exp@logacts - log_solubility_constants_exp
    reaction_vector_exp = reaction_function_exp(logsatur_exp, log_solubility_constants_exp, TK)
    transport_vector = transport_constant*(molals_bulk - molals*relative_diffusion_vector)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = reduced_balance_matrix@(transport_vector - \
                            reduced_stoich_matrix_sol_exp.transpose()@reaction_vector_exp - \
                            reduced_stoich_matrix_sol_imp.transpose()@reaction_vector_imp)
    res3 = log_solubility_constants_imp - stoich_matrix_sol_imp@logacts - stability_indexes_imp
    res4 = reaction_vector_imp*stability_indexes_imp
    res = np.hstack([res1, res2, res3, res4])
    
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    reaction_vector_prime_exp = reaction_function_derivative_exp(logsatur_exp,
                                                           log_solubility_constants_exp, TK)
    jacobian11 = -reduced_stoich_matrix*activity_hessian_diag
    jacobian12 = np.zeros((jacobian11.shape[0], nri))
    jacobian13 = np.zeros((jacobian11.shape[0], nri))
    jacobian21a = -reduced_balance_matrix*transport_constant*relative_diffusion_vector
    jacobian21b_ = (reduced_stoich_matrix_sol_exp.transpose()*reaction_vector_prime_exp)@\
                  (reduced_stoich_matrix_sol_exp*activity_hessian_diag)
    jacobian21b = -reduced_balance_matrix@jacobian21b_
    jacobian21 = jacobian21a + jacobian21b
    jacobian22 = -reduced_balance_matrix@reduced_stoich_matrix_sol_imp.transpose()
    jacobian23 = np.zeros((jacobian21.shape[0], nri))
    jacobian31 = -reduced_stoich_matrix_sol_imp*activity_hessian_diag
    jacobian32 = np.zeros((nri, nri))
    jacobian33 = -np.eye(nri)
    jacobian41 = np.zeros((nri, nn))
    jacobian42 = np.diag(stability_indexes_imp)
    jacobian43 = np.diag(reaction_vector_imp)
    jacobian = np.block([[jacobian11, jacobian12, jacobian13],
                         [jacobian21, jacobian22, jacobian23],
                         [jacobian31, jacobian32, jacobian33],
                         [jacobian41, jacobian42, jacobian43]])
    return res, jacobian