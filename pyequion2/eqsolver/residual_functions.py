# -*- coding: utf-8 -*-
import numpy as np


def residual_and_jacobian_solutes(molals,TK,activity_function,
                                  formula_vector,log_equilibrium_constants,
                                  stoich_matrix,formula_matrix):
    reduced_formula_matrix = formula_matrix[:,1:]
    reduced_stoich_matrix = stoich_matrix[:,1:]
    logacts = activity_function(molals,TK)
    res1 = log_equilibrium_constants - stoich_matrix@logacts
    res2 = reduced_formula_matrix@molals - formula_vector
    res = np.hstack([res1,res2])
    
    #Jacobian approximation
    mole_fractions = molals/(np.sum(molals))
    activity_hessian_diag = (1-mole_fractions)/molals
    upper_jacobian = -reduced_stoich_matrix*activity_hessian_diag
    lower_jacobian = reduced_formula_matrix
    jacobian = np.vstack([upper_jacobian,lower_jacobian])
    return res,jacobian