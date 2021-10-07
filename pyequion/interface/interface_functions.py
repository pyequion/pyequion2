# -*- coding: utf-8 -*-
import numpy as np


LOGE10 = 2.302585092994046


def log_decorator(f):
    def g(logsatur, logksp, *args):
        satur = 10**logsatur
        ksp = 10**logksp
        return f(satur, ksp, *args)
    return g


def log_decorator_deriv(df):
    def dg(logsatur, logksp, *args):
        satur = 10**logsatur
        ksp = 10**logksp
        return df(satur, ksp, *args)*satur*LOGE10
    return dg


def linear_ksp(satur, ksp, reaction_constant):
    return reaction_constant*ksp*(satur-1)*(satur >= 1)


def linear_ksp_deriv(satur, ksp, reaction_constant):
    return reaction_constant*ksp*(satur >= 1)


def linear(satur, ksp, reaction_constant):
    return reaction_constant*(satur-1)*(satur >= 1)


def linear_deriv(satur, ksp, reaction_constant):
    return reaction_constant*(satur >= 1)


def spinoidal(satur, ksp, reaction_constant):
    return reaction_constant*(satur**0.5-1)**2*(satur >= 1)


def spinoidal_deriv(satur, ksp, reaction_constant):
    return reaction_constant/(satur**0.5)*(satur**0.5-1)*(satur > 1)


#HINT: should be in beginning according to PEP8, but couldn't
INTERFACE_MAP = \
    {'linear_ksp': (log_decorator(linear_ksp), log_decorator_deriv(linear_ksp_deriv)),
     'linear': (log_decorator(linear), log_decorator_deriv(linear_deriv)),
     'spinoidal': (log_decorator(spinoidal), log_decorator_deriv(spinoidal_deriv))}
