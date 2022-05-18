# -*- coding: utf-8 -*-

from pyequion2 import converters

el = 'CaCO3' #
x = 1.0 #mol/kg H2O \approx mol/L
y = converters.molal_to_mgl(x, el)
x2 = converters.mgl_to_molal(y, el)
print(x)
print(y)
print(x2)