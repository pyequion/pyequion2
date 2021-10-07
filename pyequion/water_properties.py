# -*- coding: utf-8 -*-
import numpy as np

from scipy import interpolate


WATER_TEMPERATURE_CELSIUS_LIST = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                                  35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
                                  65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
                                  95.0, 100.0]  # degrees Celsius
WATER_SPECIFIC_HEAT_CAPACITY_LIST = [4200, 4200, 4188, 4184, 4183, 4183, 4183,
                                     4183, 4182, 4182, 4181, 4182, 4183,
                                     4184, 4187, 4190, 4194, 4199, 4204,
                                     4210, 4210]  # J/(kg K) 100 degrees not this
WATER_THERMAL_CONDUCTIVITY = [0.5516, 0.5516, 0.5674, 0.5769, 0.5861, 0.5948, 0.6030,
                              0.6107, 0.6178, 0.6244, 0.6305, 0.6360, 0.6410,
                              0.6455, 0.6495, 0.6530, 0.6562, 0.6589, 0.6613,
                              0.6634, 0.6634]  # W/(m K) 100 degrees not this

_water_thermal_conductivity_celsius = \
    interpolate.interp1d(WATER_TEMPERATURE_CELSIUS_LIST, WATER_THERMAL_CONDUCTIVITY,
                         fill_value="extrapolate")
_water_specific_heat_capacity_celsius = \
    interpolate.interp1d(WATER_TEMPERATURE_CELSIUS_LIST, WATER_SPECIFIC_HEAT_CAPACITY_LIST,
                         fill_value="extrapolate")


def water_density(T):  # T in Kelvin, density in kg m^3
    T = T - 273.15
    return 999.99399 + 0.04216485*T - 0.007097451*T**2 + 0.00003509571*T**3 - 9.9037785*1e-8*T**4


def water_dynamic_viscosity(T):  # T in Kelvin, returns Pa s
    return 1.856*1e-14 *\
        np.exp(4209/T + 0.04527*T + (-3.376*1e-5)*T**2)  # Pa s


def water_thermal_conductivity(T):  # T in Kelvin
    return _water_thermal_conductivity_celsius(T-273.15)[()]


def water_specific_heat_capacity(T):  # T in Kelvin
    return _water_specific_heat_capacity_celsius(T-273.15)[()]


def water_kinematic_viscosity(T): #T in Kelvin, returns m^2/s
    return water_dynamic_viscosity(T)/water_density(T)