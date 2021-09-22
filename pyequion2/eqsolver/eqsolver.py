# -*- coding: utf-8 -*-
import functools

import numpy as np

from . import residual_functions
from . import solvers


def solve_equilibrium_solutes(x_guess,
                              TK,
                              activity_function,
                              formula_vector,
                              formula_matrix,
                              stoich_matrix,
                              log_equilibrium_constants,
                              solver_function=None,
                              tol=1e-6):
    if not solver_function:
        solver_function = solvers.solver_constrained_newton
    f = functools.partial(residual_functions.residual_and_jacobian,
                          TK,
                          activity_function,
                          formula_vector,
                          formula_matrix,
                          stoich_matrix,
                          log_equilibrium_constants)
    x, res = solver_function(f, x_guess, tol=tol)
    return x, res