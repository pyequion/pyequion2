# -*- coding: utf-8 -*-
import functools


def make_solver_log(molal_balance, activity_balance,
                    molal_balance_log, activity_balance_log,
                    temperature, pressure,
                    closing_equation, closing_equation_value,
                    npoints = None):
    lines = []
    if not is_sequence(temperature):
        temperature_line = "T - {0:.2f} K".format(temperature)
    else:
        temperature_line = "T - ({0:.2f}, {1:.2f}) K".format(temperature[0], temperature[1])
    if not is_sequence(pressure):
        pressure_line = "P - {0:.2f} atm".format(pressure)
    else:
        pressure_line = "P - ({0:.2f}, {1:.2f}) atm".format(pressure[0], pressure[1])
    lines.append(temperature_line)
    lines.append(pressure_line)
    for key, value in molal_balance.items():
        if not is_sequence(value):    
            balance_line = "[{0}]={1:.2e} mol/kg H2O".format(key, value)
        else:
            balance_line = "[{0}]=({1:.2e}, {2:.2e}) mol/kg H2O".format(key, value[0], value[1])
        lines.append(balance_line)
    for key, value in activity_balance.items():
        if not is_sequence(value):
            balance_line = "{{{0}}}={1:.2e} mol/kg H2O".format(key, value)
        else:
            balance_line = "{{{0}}}=({1:.2e}, {2:.2e})  mol/kg H2O".format(key, value[0], value[1])
        lines.append(balance_line)
    for key, value in molal_balance_log.items():
        if not is_sequence(value):    
            balance_line = "log[{0}]={1:.2e} log-molal".format(key, value)
        else:
            balance_line = "log[{0}]=({1:.2e}, {2:.2e}) log-mol/kg H2O".format(key, value[0], value[1])
        lines.append(balance_line)
    for key, value in activity_balance_log.items():
        if not is_sequence(value):    
            balance_line = "log{{{0}}}={1:.2e} log-molal".format(key, value)
        else:
            balance_line = "log{{{0}}}=({1:.2e}, {2:.2e}) log-mol/kg H2O".format(key, value[0], value[1])
        lines.append(balance_line)
    if npoints is not None:
        points_line = "NPOINTS - {0}".format(npoints)
        lines.append(points_line)
    log = "\n".join(lines)
    return log


def is_sequence(obj):
    try:
        len(obj)
    except:
        return False
    else:
        return True
    
    
def is_number(obj):
    try:
        float(obj)
    except:
        False
    else:
        return True