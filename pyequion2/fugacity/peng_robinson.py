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
    kappa = 0.37464 + 1.54226*Omega - 0.26993*Omega**2 #adimensionless
    #Tr = np.clip(T/T_c, a_min=0.0, a_max=1.0) #adimensionless
    Tr = T/T_c
    alpha_ = (1 + kappa*(1 - np.sqrt(Tr)))**2 #adimensionless
    a_ = 0.45723533*_GAS_CONSTANT_ATM**2*T_c**2/P_c #atm cm^6 mol^{-2}
    b_ = 0.07779607*_GAS_CONSTANT_ATM*T_c/P_c #cm^3/mol
    a_alpha_ = a_*alpha_
    b = np.sum(x*b_)
    a_alpha = np.sum((x*x[..., None])*np.sqrt(a_alpha_*a_alpha_[..., None]))
    return a_alpha, b


def _peng_robinson_comprehensibility(a_alpha, b, T, P):
    A = a_alpha*P/(_GAS_CONSTANT_ATM**2*T**2)
    B = b*P/(_GAS_CONSTANT_ATM*T)
    C3 = 1
    C2 = B - 1
    C1 = A - 2*B - 3*B**2
    C0 = B**3 + B**2 - A*B
    roots = solve_cubic.solve_cubic(C3, C2, C1, C0)
    Z = np.real(roots[0])  #Surely real (biggest)
    return Z


def _peng_robinson_fugacity(x, T, P, T_c, P_c, Omega): #x: molal_fractions, T: K, P: atm 
    a_alpha, b = _peng_robinson_a_and_b(x, T, P, T_c, P_c, Omega)
    comprehensibility = _peng_robinson_comprehensibility(a_alpha, b, T, P) #P*V_m/(R*T)
    const1 = 2*np.sqrt(2)
    const2 = 1 + np.sqrt(2)
    const3 = np.sqrt(2) - 1
    V_m = comprehensibility*(_GAS_CONSTANT_ATM*T)/P
    logphi1 = P*V_m/(_GAS_CONSTANT_ATM*T) - 1
    logphi2 = -np.log(P*(V_m-b)/(_GAS_CONSTANT_ATM*T))
    logphi3a = a_alpha/(const1*b*(_GAS_CONSTANT_ATM*T))
    logphi3b = np.log((V_m + const2*b)/(V_m - const3*b))
    logphi3 = -logphi3a*logphi3b
    logphi = logphi1 + logphi2 + logphi3 #base3
    logphi = logphi/2.302585092994046 #base e to base 10
    return logphi