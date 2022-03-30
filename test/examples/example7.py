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


elements = ['Ca', 'C', 'Mg']
intsys = InterfaceSystem(elements, from_elements=True)
intsys.set_interface_phases(['Calcite', 'Dolomite'])

TK = 298.15
pipe_diameter = 0.01 #m
flow_velocity = 1.0
pipe_length = 80.0 #m
pipe_time = pipe_length/flow_velocity

co2_flash_value = 0.001
initial_ca_value = 0.02
initial_mg_value = 0.01

transport_params = {'type': 'pipe',
                    'shear_velocity': shear_velocity(flow_velocity, pipe_diameter, TK)}
solution_stats = {'res': None, 'x': 'default'}
solution_stats_int = {'res': None, 'x': 'default'}

def f(t, y):
    global solution_stats
    global solution_stats_int
    molal_balance = {'Ca': y[0], 'Mg': y[1], 'CO2': co2_flash_value}
    solution, solution_stats = intsys.solve_equilibrium_mixed_balance(TK,
                                                                      molal_balance=molal_balance,
                                                                      tol=1e-6,
                                                                      initial_guess=solution_stats['x'])
    molals_bulk = solution.solute_molals
    solution_int, solution_stats_int = intsys.solve_interface_equilibrium(TK,
                                                                          molals_bulk,
                                                                          transport_params,
                                                                          tol=1e-6,
                                                                          initial_guess=solution_stats_int['x'])
    elements_reaction_fluxes = solution_int.elements_reaction_fluxes
    wall_scale = 4/(pipe_diameter*water_properties.water_density(TK))
    dy = -wall_scale*np.array(
        [elements_reaction_fluxes['Ca'], elements_reaction_fluxes['Mg']])
    return dy


initial_vector = np.array([initial_ca_value, initial_mg_value])
start_time = time.time()
sol = scipy.integrate.solve_ivp(f, (0.0, pipe_time), initial_vector)
elapsed_time = time.time() - start_time
