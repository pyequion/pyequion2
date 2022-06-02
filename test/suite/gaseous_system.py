# -*- coding: utf-8 -*-

import pyequion2
from pyequion2 import gaseous_system, converters

gsys = gaseous_system.InertGaseousSystem(["CO2(g)"], fugacity_model="PENGROBINSON")
logacts = gsys.get_fugacity({"CO2(g)":1.0}, 300, 100)
fugacity = 10**logacts['CO2(g)']