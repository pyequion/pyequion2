# -*- coding: utf-8 -*-
import functools

import numpy as np

from . import solve_cubic


_GAS_CONSTANT_ATM = 82.06 #atm cm^3 mol^{-1} K^{-1}


def make_peng_robinson_fugacity_function(reactions_gases):
    T_c, P_c, Omega = _get_peng_robinson_params(reactions_gases)
    f = functools.partial(_peng_robinson_fugacity, T_c=T_c, P_c=P_c, Omega=Omega)
    return f


def _get_peng_robinson_params(reactions_gases):
    T_c = np.array([r.get('T_c', 0.0) for r in reactions_gases])
    P_c = np.array([r.get('P_c', 1.0) for r in reactions_gases])
    Omega = np.array([r.get('Omega', 0.0) for r in reactions_gases])
    return T_c, P_c, Omega


def _peng_robinson_a_and_b(x, T, P, T_c, P_c, Omega):
    kappa = 0.37464 + 1.54226*Omega - 0.26992*Omega**2 #adimensionless
    Tr = T/T_c #adimensionless
    alpha_ = (1 + kappa*(1 - np.sqrt(Tr)))**2 #adimensionless
    a_ = 0.45724*_GAS_CONSTANT_ATM**2*T_c**2/P_c #atm cm^2 mol^{-2}
    b_ = 0.07780*_GAS_CONSTANT_ATM*T_c/P_c #cm^3/mol
    a_alpha_ = a_*alpha_
    b = np.sum(x*b_)
    a_alpha = np.sum((x*x[..., None])*np.sqrt(a_alpha_*a_alpha_[..., None]))
    return a_alpha, b


def _peng_robinson_comprehensibility(a_alpha, b, T, P):
    A = a_alpha*P/(_GAS_CONSTANT_ATM**2*T**2)
    B = b*P/(_GAS_CONSTANT_ATM*T)
    roots = solve_cubic.solve_cubic(1, -(1-B), (A-2*B-3*B**2), -(A*B - B**2 - B**3))
    comprehensibility = np.real(roots[0])  #Surely real (biggest)
    return comprehensibility


def _peng_robinson_fugacity(x, T, P, T_c, P_c, Omega): #x: molal_fractions, T: K, P: atm 
    a_alpha, b = _peng_robinson_a_and_b(x, T, P, T_c, P_c, Omega)
    comprehensibility = _peng_robinson_comprehensibility(a_alpha, b, T, P) #P*V_m/(R*T)
    V_m = comprehensibility*(_GAS_CONSTANT_ATM*T)/P
    logphi1 = P*V_m/(_GAS_CONSTANT_ATM*T) - 1
    logphi2 = np.log(P*(V_m-b)/(_GAS_CONSTANT_ATM*T))
    logphi3 = a_alpha/(2.828*b*(_GAS_CONSTANT_ATM*T))*np.log((V_m + 2.414*b)/(V_m - 0.414*b))
    logphi = logphi1 + logphi2 + logphi3
    return logphi