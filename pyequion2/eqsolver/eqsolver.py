# -*- coding: utf-8 -*-
import functools

import numpy as np

from . import residual_functions
from . import solvers


def solve_equilibrium_solutes(x_guess,
                              TK,
                              activity_function,
                              balance_vector_molal,
                              balance_vector_logact,
                              log_equilibrium_constants,
                              balance_matrix_molal,
                              balance_matrix_logact,
                              stoich_matrix,
                              solver_function=None,
                              tol=1e-6):
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    f = functools.partial(residual_functions.residual_and_jacobian_solutes,
                          TK=TK,
                          activity_function=activity_function,
                          balance_vector_molal=balance_vector_molal,
                          balance_vector_logact=balance_vector_logact,
                          log_equilibrium_constants=log_equilibrium_constants,
                          balance_matrix_molal=balance_matrix_molal,
                          balance_matrix_logact=balance_matrix_logact,
                          stoich_matrix=stoich_matrix)
    x, res = solver_function(f, x_guess, tol=tol)
    return x, res