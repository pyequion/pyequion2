# -*- coding: utf-8 -*-

#DEPRECATED: Use pyequion.logmaker.py


def make_solver_log(molal_balance, activity_balance,
                    molal_balance_log, activity_balance_log,
                    temperature, pressure,
                    closing_equation, closing_equation_value):
    lines = []
    temperature_line = "T - {0:.2f} atm".format(temperature)
    pressure_line = "P - {0:.2f} atm".format(pressure)
    lines.append(temperature_line)
    lines.append(pressure_line)
    for key, value in molal_balance.items():
        balance_line = "[{0}]={1:.2e} molal".format(key, value)
        lines.append(balance_line)
    for key, value in activity_balance.items():
        balance_line = "{{{0}}}={1:.2e} molal".format(key, value)
        lines.append(balance_line)
    for key, value in molal_balance_log.items():
        balance_line = "log[{0}]={1:.2e} molal".format(key, value)
        lines.append(balance_line)
    for key, value in activity_balance_log.items():
        balance_line = "log{{{0}}} - {1:.2e} molal".format(key, value)
        lines.append(balance_line)
    log = "\n".join(lines)
    return log