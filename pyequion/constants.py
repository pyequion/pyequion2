# -*- coding: utf-8 -*-

MOLAR_WEIGHT_WATER = 18.01528*1e-3 #kg/mol
LOG10E = 0.4342944819032518
B_DEBYE = 1.2 #kg^0.5 mole^-0.5
ALKALINE_COEFFICIENTS = {
        'HCO3-': 1.0,
        'CO3--': 2.0,
        'OH-': 1.0,
        'HPO4--': 1.0,
        'PO4---': 2.0,
        'H3SiO4--': 1.0,
        'NH3': 1.0,
        'HS-': 1.0,
        'H+': -1.0,
        'HSO4--': -1.0,
        'HF': -1.0,
        'H3PO4': -1.0,
        'HNO2': -1.0
        }
TURBULENT_VISCOSITY_CONSTANT = 9.5*1e-4
PATM = 101325.
PBAR = 1e5