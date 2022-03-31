import numpy as np
import cython


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
def coo_matrix_vector(double[:] A_data, int[:,:] A_inds, int[:] A_shape, double[:] b):
    assert A_shape[1] == b.shape[0]
    result = np.zeros(A_shape[0],dtype=np.double)
    cdef double[:] result_view = result
    cdef Py_ssize_t i
    cdef Py_ssize_t imax = A_data.shape[0]
    
    cdef int ind0,ind1
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        result_view[ind0] += A_data[i]*b[ind1]
    return result


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
def coo_matrix_vector_vector(double[:] A_data, int[:,:] A_inds, int[:] A_shape,
                             double[:] b1, double[:] b2):
    assert A_shape[0] == b1.shape[0]
    assert A_shape[1] == b2.shape[0]
        
    cdef double result = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t imax = A_data.shape[0]
    
    cdef int ind0,ind1
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        result += A_data[i]*b1[ind0]*b2[ind1]
    return result


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
def coo_tensor_vector_vector(double[:] A_data, int[:,:] A_inds, int[:] A_shape,
                             double[:] b1, double[:] b2):
    assert A_shape[1] == b1.shape[0]
    assert A_shape[2] == b2.shape[0]
    
    result = np.zeros(A_shape[0],dtype=np.double)
    
    cdef double[:] result_view = result
    cdef Py_ssize_t i
    cdef Py_ssize_t imax = A_data.shape[0]
    
    cdef int ind0,ind1,ind2
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        ind2 = A_inds[i,2]
        result_view[ind0] += A_data[i]*b1[ind1]*b2[ind2]
    return result

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
def coo_tensor_vector_vector_vector(double[:] A_data, int[:,:] A_inds, int[:] A_shape,
                                    double[:] b1, double[:] b2, double[:] b3):
    assert A_shape[0] == b1.shape[0]
    assert A_shape[1] == b2.shape[0]
    assert A_shape[2] == b3.shape[0]
        
    cdef double result = 0.0
    cdef Py_ssize_t i
    cdef Py_ssize_t imax = A_data.shape[0]
    
    cdef int ind0, ind1, ind2
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        ind2 = A_inds[i,2]
        result += A_data[i]*b1[ind0]*b2[ind1]*b3[ind2]
    return result
