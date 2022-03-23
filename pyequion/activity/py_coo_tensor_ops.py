import numpy as np


def coo_matrix_vector(A_data, A_inds, A_shape, b):
    assert A_shape[1] == b.shape[0]
    result = np.zeros(A_shape[0],dtype=np.double)
    imax = A_data.shape[0]
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        result[ind0] += A_data[i]*b[ind1]
    return result


def coo_matrix_vector_vector(A_data, A_inds, A_shape,
                             b1, b2):
    assert A_shape[0] == b1.shape[0]
    assert A_shape[1] == b2.shape[0]
    imax = A_data.shape[0]
    result = 0.0
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        result += A_data[i]*b1[ind0]*b2[ind1]
    return result


def coo_tensor_vector_vector(A_data, A_inds, A_shape,
                             b1, b2):
    assert A_shape[1] == b1.shape[0]
    assert A_shape[2] == b2.shape[0]
    
    result = np.zeros(A_shape[0],dtype=np.double)
    imax = A_data.shape[0]
    
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        ind2 = A_inds[i,2]
        result[ind0] += A_data[i]*b1[ind1]*b2[ind2]
    return result

# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)   # Deactivate negative indexing.
def coo_tensor_vector_vector_vector(A_data, A_inds, A_shape,
                                    b1, b2, b3):
    assert A_shape[0] == b1.shape[0]
    assert A_shape[1] == b2.shape[0]
    assert A_shape[2] == b3.shape[0]
    imax = A_data.shape[0]
    result = 0.0        
    for i in range(imax):
        ind0 = A_inds[i,0]
        ind1 = A_inds[i,1]
        ind2 = A_inds[i,2]
        result += A_data[i]*b1[ind0]*b2[ind1]*b3[ind2]
    return result
