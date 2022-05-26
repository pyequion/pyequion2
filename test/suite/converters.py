# -*- coding: utf-8 -*-

from pyequion2 import converters

def test1():
    el = 'CaCO3' #
    x = 1.0 #mol/kg H2O \approx mol/L
    y = converters.molal_to_mgl(x, el)
    CORRECT_VALUE_Y = 99798.8386920672
    assert(abs(y-CORRECT_VALUE_Y)/CORRECT_VALUE_Y < 1e-6)
    x2 = converters.mgl_to_molal(y, el)
    assert(abs(x - x2)/x2 < 1e-6)

def test2():
    spec = 'CO2'
    pp = 1.0
    TK = 298.15
    act = converters.get_activity_from_partial_pressure(pp, spec, TK)
    comparisor = 3.4*1e-2
    assert(abs(act - comparisor)/comparisor < 1e-2)
    
def test3():
    phase = "Calcite"
    molar_mass = converters.phase_to_molar_weight("Calcite")
    print(molar_mass)

test3()