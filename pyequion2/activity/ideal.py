# -*- coding: utf-8 -*-
import warnings

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except (ImportError, AssertionError):
    warnings.warn("JAX not installed. Only numpy can be chosen as backend")

from .. import constants

def setup_ideal(solutes, calculate_osmotic_coefficient=False, backend='numpy'):
    assert backend in ['numpy', 'jax']
    if backend == 'numpy':
        def g(x, TK):
            return np.insert(np.zeros_like(x), 0, constants.LOG10E)
    else:
        def g(x, TK):
            return jnp.insert(jnp.zeros_like(x), 0, constants.LOG10E)     
        g = jax.jit(g)
    return g
