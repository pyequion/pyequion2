# -*- coding: utf-8 -*-
import numpy as np

from .. import constants


def setup_ideal(solutes, calculate_osmotic_coefficient=False):
    g = lambda x,TK : np.insert(np.zeros_like(x),0,constants.LOG10E)
    return g
