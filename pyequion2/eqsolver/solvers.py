# -*- coding: utf-8 -*-
import numpy as np


def solver_constrained_newton(f, x0, maxiter=10000, tol=1e-6,
                              delta_step=0.9999,
                              max_step=1.0,
                              print_frequency=None):
    delta_step = 0.9999
    control_value = 10**-200
    x = x0.copy()
    for i in range(maxiter):
        res, jac = f(x)
        if np.max(np.abs(res)) < tol:
            break
        try:
            delta_x = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            print("Solver jacobian is singular. Returning value and residual as it is")
            return x, res
        control_index = np.abs(delta_x) > control_value
        x_step = x[control_index]
        delta_x_step = delta_x[control_index]
        step_ = -delta_step*x_step/delta_x_step*((x_step + delta_x_step) <= 0) + \
            max_step*(x_step+delta_x_step > 0)
        step = np.min(step_)
        x_new = x + step*delta_x
        x_new[x_new < control_value] = control_value
        if print_frequency is not None:
            if (i+1) % print_frequency == 0:
                print('------')
                print(x)
                print(res)
                print(i)
                print('------')
        x = x_new
    return x, res
