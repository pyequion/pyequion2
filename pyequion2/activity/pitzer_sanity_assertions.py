# -*- coding: utf-8 -*-
import numpy as np


def make_sanity_assertions():
    from .coo_tensor_ops import coo_tensor_ops
    from . import py_coo_tensor_ops
    
    for i in range(2):
        n = 50 + 20*i
        A_data = np.random.randn(n).astype(np.double)
        A_inds = np.random.randint(n, size=(n,2), dtype=np.intc)
        A_shape = np.array([n, n], dtype=np.intc)
        At_data = np.random.randn(n).astype(np.double)
        At_inds = np.random.randint(n, size=(n,3), dtype=np.intc)
        At_shape = np.array([n, n, n], dtype=np.intc)
        b1 = np.random.randn(n).astype(np.double)
        b2 = np.random.randn(n).astype(np.double)
        b3 = np.random.randn(n).astype(np.double)
        
        res_cython_1 = coo_tensor_ops.coo_matrix_vector(A_data, A_inds, A_shape, b1)
        res_python_1 = py_coo_tensor_ops.coo_matrix_vector(A_data, A_inds, A_shape, b1)
        res_cython_2 = coo_tensor_ops.coo_matrix_vector_vector(A_data, A_inds, A_shape, b1, b2)
        res_python_2 = py_coo_tensor_ops.coo_matrix_vector_vector(A_data, A_inds, A_shape, b1, b2)
        res_cython_3 = coo_tensor_ops.coo_tensor_vector_vector(At_data, At_inds, At_shape, b1, b2)
        res_python_3 = py_coo_tensor_ops.coo_tensor_vector_vector(At_data, At_inds, At_shape, b1, b2)
        res_cython_4 = coo_tensor_ops.coo_tensor_vector_vector_vector(At_data, At_inds, At_shape, b1, b2, b3)
        res_python_4 = py_coo_tensor_ops.coo_tensor_vector_vector_vector(At_data, At_inds, At_shape, b1, b2, b3)
        
        
        assert(np.max(np.abs(res_cython_1 - res_python_1)) < 1e-12)
        assert(np.max(np.abs(res_cython_2 - res_python_2)) < 1e-12)
        assert(np.max(np.abs(res_cython_3 - res_python_3)) < 1e-12)
        assert(np.max(np.abs(res_cython_4 - res_python_4)) < 1e-12)