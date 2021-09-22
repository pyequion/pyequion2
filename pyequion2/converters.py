# -*- coding: utf-8 -*-

from . import water_properties


def mmolar_to_molal(x,TK=298.15):
    return x/water_properties.water_density(TK)


def molal_to_mmolar(x,TK=298.15):
    return x*water_properties.water_density(TK)