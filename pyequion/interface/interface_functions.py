# -*- coding: utf-8 -*-
import numpy as np


LOGE10 = 2.302585092994046
GAS_CONSTANT = 8.314462618

def log_decorator(f):
    def g(logsatur, logksp, TK, *args):
        satur = 10**logsatur
        ksp = 10**logksp
        return f(satur, ksp, TK, *args)
    return g


def log_decorator_deriv(df):
    def dg(logsatur, logksp, TK, *args):
        satur = 10**logsatur
        ksp = 10**logksp
        return df(satur, ksp, TK, *args)*satur*LOGE10
    return dg


def arrhenize(f):
    def f_(satur, ksp, TK, preexp, energy):
        reaction_constant = preexp*np.exp(-energy/(GAS_CONSTANT*TK))
        return f(satur, ksp, TK, reaction_constant)
    return f_

        
def linear_ksp(satur, ksp, TK, reaction_constant):
    return reaction_constant*ksp*(satur-1)*(satur >= 1)


def linear_ksp_deriv(satur, ksp, TK, reaction_constant):
    return reaction_constant*ksp*(satur >= 1)


def linear(satur, ksp, TK, reaction_constant):
    return reaction_constant*(satur-1)*(satur >= 1)


def linear_deriv(satur, ksp, TK, reaction_constant):
    return reaction_constant*(satur >= 1)


def spinoidal(satur, ksp, TK, reaction_constant):
    return reaction_constant*(satur**0.5-1)**2*(satur >= 1)


def spinoidal_deriv(satur, ksp, TK, reaction_constant):
    return reaction_constant/(satur**0.5)*(satur**0.5-1)*(satur > 1)


#HINT: should be in beginning according to PEP8, but couldn't
INTERFACE_MAP = \
    {'linear_ksp': (log_decorator(linear_ksp), log_decorator_deriv(linear_ksp_deriv)),
     'linear': (log_decorator(linear), log_decorator_deriv(linear_deriv)),
     'spinoidal': (log_decorator(spinoidal), log_decorator_deriv(spinoidal_deriv)),
     'linear_ksp_temp': (log_decorator(arrhenize(linear_ksp)), log_decorator_deriv(arrhenize(linear_ksp_deriv))),
     'linear_temp': (log_decorator(arrhenize(linear)), log_decorator_deriv(arrhenize(linear_deriv))),
     'spinoidal_temp': (log_decorator(arrhenize(spinoidal)), log_decorator_deriv(arrhenize(spinoidal_deriv)))}
    
    
SPECIFIC_SOLIDS_MODEL = \
    {'Calcite': ('linear_ksp_temp', (8.673367178929761e+19, 86881.05))}
