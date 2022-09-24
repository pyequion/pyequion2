# -*- coding: utf-8 -*-
import warnings

import numpy as np
try:
    import jax
    import jax.numpy as jnp
except (ImportError, AssertionError):
    warnings.warn("JAX not installed, so can't be used as backend")
try:
    import torch
except (ImportError, AssertionError):
    warnings.warn("PyTorch not installed, so can't be used as backend")
    
from .. import constants

def setup_ideal(solutes, calculate_osmotic_coefficient=False, backend='numpy'):
    assert backend in ['numpy', 'jax', "torch"]
    if backend == 'numpy':
        def g(x, TK):
            constant = np.ones(x.shape[:-1])*constants.LOG10E
            return np.hstack([constant, np.zeros_like(x)])
    elif backend == "jax":
        def g(x, TK):
            constant = jnp.ones(x.shape[:-1])*constants.LOG10E
            return jnp.hstack([constant, jnp.zeros_like(x)])
        g = jax.jit(g)
    elif backend == "torch":
        def g(x, TK):
            return torch.cat([x[..., :1]*constants.LOG10E, x*0.0], dim=-1)
    return g
