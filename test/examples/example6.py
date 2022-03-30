# -*- coding: utf-8 -*-
import time

import numpy as np
import scipy.integrate

from pyequion2 import InterfaceSystem
from pyequion2 import water_properties


def reynolds_number(flow_velocity, pipe_diameter, TK=298.15): #Dimensionless
    kinematic_viscosity = water_properties.water_kinematic_viscosity(TK)
    return flow_velocity*pipe_diameter/kinematic_viscosity


def darcy_friction_factor(flow_velocity, pipe_diameter, TK=298.15):
    reynolds = reynolds_number(flow_velocity, pipe_diameter, TK)
    if reynolds < 2300:
        return 64/reynolds
    else: #Blasius
        return 0.316*reynolds**(-1./4)
    

def shear_velocity(flow_velocity, pipe_diameter, TK=298.15):
    f = darcy_friction_factor(flow_velocity, pipe_diameter, TK)
    return np.sqrt(f/8.0)*flow_velocity


elements = ['Ca', 'C', 'Na', 'Cl', 'Mg']
intsys = InterfaceSystem(elements, from_elements=True)
intsys.set_interface_phases(['Calcite', 'Dolomite'])
index_map = {el: i for i, el in enumerate(elements)}
reverse_index_map = {i: el for i, el in enumerate(elements)}

TK = 298.15
pipe_diameter = 0.05 #m
flow_velocity = 1.0
pipe_length = 80.0 #m
pipe_time = pipe_length/flow_velocity

transport_params = {'type': 'pipe',
                    'shear_velocity': shear_velocity(flow_velocity, pipe_diameter, TK)}
solution_stats = {'res': None, 'x': 'default'}
solution_stats_int = {'res': None, 'x': 'default'}

def f(t, y):
    global solution_stats
    global solution_stats_int
    elements_balance = {el: y[index_map[el]] for el in elements}
    solution, solution_stats = intsys.solve_equilibrium_elements_balance(TK,
                                                                         elements_balance,
                                                                         tol=1e-6,
                                                                         initial_guess=solution_stats['x'])
    molals_bulk = solution.solute_molals
    solution_int, solution_stats_int = intsys.solve_interface_equilibrium(TK,
                                                                          molals_bulk,
                                                                          transport_params,
                                                                          tol=1e-6,
                                                                          initial_guess=solution_stats_int['x'])
    elements_reaction_fluxes = solution_int.elements_reaction_fluxes
    dy = -4/pipe_diameter*np.hstack(
        [elements_reaction_fluxes[reverse_index_map[i]]
         for i in range(y.shape[0])])
    return dy


initial_elements_balance = {'Ca':0.028, 'C':0.065, 'Na':0.075, 'Cl':0.056, 'Mg':0.02}
initial_elements_vector = np.hstack([initial_elements_balance[reverse_index_map[i]]
                                     for i in range(len(initial_elements_balance))])

start_time = time.time()
sol = scipy.integrate.solve_ivp(f, (0.0, pipe_time), initial_elements_vector)
elapsed_time = time.time() - start_time
