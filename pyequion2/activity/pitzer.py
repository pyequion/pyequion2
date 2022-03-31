# -*- coding: utf-8 -*
import os
import pathlib
import re
import functools
import warnings

import numpy as np

try:
    from .pitzer_sanity_assertions import make_sanity_assertions
    make_sanity_assertions()
    from .coo_tensor_ops import coo_tensor_ops
    from . import py_coo_tensor_ops
except (ImportError, AssertionError): #Some import error. Use pythonized way
    warnings.warn("Problem with Cython import. Using pure python operation.")
    from . import py_coo_tensor_ops as coo_tensor_ops
from .. import utils
from .. import constants
from .. import datamods

    
def setup_pitzer(solutes, calculate_osmotic_coefficient=False):
    property_dict = make_pitzer_dictionary()
    B0, B0_inds = make_parameter_matrix(solutes, 'B0', property_dict)
    B1, B1_inds = make_parameter_matrix(solutes, 'B1', property_dict)
    B2, B2_inds = make_parameter_matrix(solutes, 'B2', property_dict)
    C0, C0_inds = make_parameter_matrix(solutes, 'C0', property_dict)
    THETA, THETA_inds = make_parameter_matrix(solutes, 'THETA', property_dict)
    LAMBDA, LAMBDA_inds = make_parameter_matrix(
        solutes, 'LAMDA', property_dict)
    PSI, PSI_inds = make_parameter_3_tensor(solutes, 'PSI', property_dict)
    zarray = np.array([utils.charge_number(specie) for specie in solutes],
                      dtype=np.double)
    f = functools.partial(loggamma_and_osmotic, zarray=zarray,
                          calculate_osmotic_coefficient=calculate_osmotic_coefficient,
                          B0_=B0, B0_inds=B0_inds,
                          B1_=B1, B1_inds=B1_inds,
                          B2_=B2, B2_inds=B2_inds,
                          C0_=C0, C0_inds=C0_inds,
                          THETA_=THETA, THETA_inds=THETA_inds,
                          PSI_=PSI, PSI_inds=PSI_inds,
                          LAMBDA_=LAMBDA, LAMBDA_inds=LAMBDA_inds)

    def g(xarray, TK):
        # ln(gamma) to log10(gamma)
        return constants.LOG10E * f(xarray, TK)
    return g


def loggamma_and_osmotic(carray, T, zarray,
                         calculate_osmotic_coefficient,
                         B0_, B0_inds,
                         B1_, B1_inds,
                         B2_, B2_inds,
                         C0_, C0_inds,
                         THETA_, THETA_inds,
                         PSI_, PSI_inds,
                         LAMBDA_, LAMBDA_inds):
    temp_vector = temperature_vector(T)
    B0 = np.sum(temp_vector*B0_, axis=-1)
    B1 = np.sum(temp_vector*B1_, axis=-1)
    B2 = np.sum(temp_vector*B2_, axis=-1)
    C0 = np.sum(temp_vector*C0_, axis=-1)
    THETA = np.sum(temp_vector*THETA_, axis=-1)
    PSI = np.sum(temp_vector*PSI_, axis=-1)
    LAMBDA = np.sum(temp_vector*LAMBDA_, axis=-1)

    # We are excluding ZETA here
    dim_matrices = np.array([carray.shape[0], carray.shape[0]], dtype=np.intc)
    dim_tensors = np.array(
        [carray.shape[0], carray.shape[0], carray.shape[0]], dtype=np.intc)
    if carray.dtype != np.double:
        carray = carray.astype(np.double)
    if zarray.dtype != np.double:
        zarray = zarray.astype(np.double)
    I = 0.5*np.sum(carray*zarray**2)
    sqrtI = np.sqrt(I)
    Z = np.sum(carray*np.abs(zarray))
    A = A_debye(T)

    # (1,1),(2,1) < 4, (2,2) = 4, (3,2),(4,2) > 4
    valence_prod_1 = -1*zarray[B1_inds[:, 0]]*zarray[B1_inds[:, 1]]
    valence_prod_2 = -1*zarray[B2_inds[:, 0]]*zarray[B2_inds[:, 1]]
    alpha1 = 2.0*(valence_prod_1 != 4) + 1.4*(valence_prod_1 == 4)
    alpha2 = 12.0*(valence_prod_2 <= 4) + 50.0*(valence_prod_2 > 4)

    x_mn = 6*A*zarray[THETA_inds[:, 0]]*zarray[THETA_inds[:, 1]]*sqrtI
    x_mm = 6*A*zarray[THETA_inds[:, 0]]*zarray[THETA_inds[:, 0]]*sqrtI
    x_nn = 6*A*zarray[THETA_inds[:, 1]]*zarray[THETA_inds[:, 1]]*sqrtI
    J = jtheta(x_mn) - 0.5*jtheta(x_mm) - 0.5*jtheta(x_nn)
    J_prime = (x_mn*jprime(x_mn) - 0.5*x_mm *
               jprime(x_mm) - 0.5*x_nn*jprime(x_nn))/(2*I)
    K1_theta = zarray[THETA_inds[:, 0]]*zarray[THETA_inds[:, 1]]/(4*I)
    K2_theta = K1_theta/I
    THETA_e = K1_theta*J
    PHI = THETA + THETA_e
    PHI_prime = K1_theta*J_prime - K2_theta*J  # THETA_e_prime

    C = C0/(2*np.sqrt(-1*zarray[C0_inds[:, 0]]*zarray[C0_inds[:, 1]]))

    F_1 = A*f_debye(sqrtI)
    F_21 = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
        B1*gprime(alpha1*sqrtI)/I, B1_inds, dim_matrices, carray, carray)
    F_21_py = 0.5*py_coo_tensor_ops.coo_matrix_vector_vector(
        B1*gprime(alpha1*sqrtI)/I, B1_inds, dim_matrices, carray, carray)
    F_22 = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
        B2*gprime(alpha2*sqrtI)/I, B2_inds, dim_matrices, carray, carray)
    F_31 = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
        PHI_prime, THETA_inds, dim_matrices, carray, carray)
    F = F_1 + F_21 + F_22 + F_31
    res1 = zarray**2*F

    sum_11 = 2*coo_tensor_ops.coo_matrix_vector(
        B0, B0_inds, dim_matrices, carray)
    sum_12 = 2*coo_tensor_ops.coo_matrix_vector(
        B1*gb(alpha1*sqrtI), B1_inds, dim_matrices, carray)
    sum_13 = 2*coo_tensor_ops.coo_matrix_vector(
        B2*gb(alpha2*sqrtI), B2_inds, dim_matrices, carray)
    sum_21 = Z*coo_tensor_ops.coo_matrix_vector(
        C, C0_inds, dim_matrices, carray)
    sum_22 = 0.5*np.abs(zarray) *\
        coo_tensor_ops.coo_matrix_vector_vector(
        C, C0_inds, dim_matrices, carray, carray)
    res2 = sum_11 + sum_12 + sum_13 + sum_21 + sum_22

    sum_31 = 2*coo_tensor_ops.coo_matrix_vector(
        PHI, THETA_inds, dim_matrices, carray)
    sum_32 = 0.5*coo_tensor_ops.coo_tensor_vector_vector(
        PSI, PSI_inds, dim_tensors, carray, carray)
    res3 = sum_31 + sum_32
    sum_41 = 2*coo_tensor_ops.coo_matrix_vector(
        LAMBDA, LAMBDA_inds, dim_matrices, carray)
    res4 = sum_41
    logg = res1 + res2 + res3 + res4  # res1 + res2 + res3 + res4
    #B + I*Bprime
    #B = B0 + B1*g(alpha1*sqrtI) + B2*g(alpha*sqrtI)
    #Bprime = (B1*gprime(alpha1*sqrtI) + B2*gprime(alpha2*sqrtI))/I
    # B + I*Bprime = B0 + B1*(g(alpha1*sqrtI) + gprime(alpha1*sqrtI))
    #                  + B2*(g(alpha2*sqrtI) + gprime(alpha2*sqrtI))
    # Water activity
    if not calculate_osmotic_coefficient:
        osmotic_coefficient = 1.0
    else:
        res1w = -A*sqrtI**3/(1 + constants.B_DEBYE*sqrtI)

        sum_11w = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
            B0, B0_inds, dim_matrices, carray, carray)
        sum_12w = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
            B1*(gb(alpha1*sqrtI) + gprime(alpha1*sqrtI)), B0_inds, dim_matrices, carray, carray)
        sum_13w = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
            B2*(gb(alpha2*sqrtI) + gprime(alpha2*sqrtI)), B0_inds, dim_matrices, carray, carray)
        sum_21w = 0.5*Z*coo_tensor_ops.coo_matrix_vector_vector(
            C, C0_inds, dim_matrices, carray, carray)
        res2w = sum_11w + sum_12w + sum_13w + sum_21w

        sum_31w = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
            PHI + I*PHI_prime, THETA_inds, dim_matrices, carray, carray)
        sum_32w = 1/6*coo_tensor_ops.coo_tensor_vector_vector_vector(
            PSI, PSI_inds, dim_tensors, carray, carray, carray)
        res3w = sum_31w + sum_32w
        sum_41 = 0.5*coo_tensor_ops.coo_matrix_vector_vector(
            LAMBDA, LAMBDA_inds, dim_matrices, carray, carray)
        res4w = sum_41

        resw = 2/np.sum(carray)*(res1w + res2w + res3w + res4w)
        osmotic_coefficient = (resw + 1)
    logg = np.insert(logg, 0, osmotic_coefficient)
    return logg


def A_debye(T):
    Na = 6.0232e23
    ee = 4.8029e-10
    k = 1.38045e-16
    ds = -0.0004 * T + 1.1188
    eer = 305.7 * np.exp(-np.exp(-12.741 + 0.01875 * T) - T / 219.0)
    Aphi = 1.0/3.0*(2.0 * np.pi * Na * ds / 1000) ** 0.5 * \
        (ee / (eer * k * T) ** 0.5) ** 3.0
    return Aphi


def gprime(x):
    return -2*(1-(1+x+x**2/2)*np.exp(-x))/(x**2)


def jprime(x):
    a, b, c, d = 4.581, 0.7237, 0.0120, 0.528
    return ((4+a/(x**b)*np.exp(c*x**d)) -
            (x**(1-2*b)*a*np.exp(c*x**d)*(c*d*x**(b+d-1)-b*x**(b-1)))) / \
        ((4+a*np.exp(c*x**d)/(x**b))**2)


def gb(x):
    return 2*(1-(1+x)*np.exp(-x))/(x**2)


def jtheta(x):
    return x/(4 + 4.581/(x**(0.7237))*np.exp(0.0120*x**(0.528)))


def f_debye(sqrtI):
    res = -(sqrtI/(1+constants.B_DEBYE*sqrtI) +
            2/constants.B_DEBYE*np.log(1 + constants.B_DEBYE*sqrtI))
    return res


def make_pitzer_dictionary():
    # ownpath = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    # filepath = ownpath.parents[0]/'data'/'pitzer.txt'
    # with open(filepath, 'r') as file:
    #     lines = file.read().split('\n')
    lines = datamods.pitzer_data.split('\n')
    # Excluding (OH) labeled elements (boron elements, essentialy) and PITZER line
    # lines = [line for line in lines[1:] if '(OH)' not in line]
    # Excluding PITZER line
    lines = lines[1:]
    lines_processed = [_process_line_pitzer(line) for line in lines]
    property_names = []
    property_indexes = []
    for i, line in enumerate(lines_processed):
        if len(line) == 1:
            property_names.append(line[0][1:])
            property_indexes.append(i)
    property_dict = dict()
    i_low = 0
    i_max = len(property_names) - 1
    for j, name in enumerate(property_names):
        if j < i_max:
            i_high = property_indexes[j+1]
            lines_processed_i = lines_processed[i_low+1:i_high]
            i_low = i_high
    #         property_dict[name] = lines_processed_i
        else:
            lines_processed_i = lines_processed[i_low+1:]
    #         property_dict[name] = lines_processed_i
        property_dict_i = dict()
        for line in lines_processed_i:
            value = line[-6:]
            key = tuple(sorted(line[:-6]))
            property_dict_i[key] = value
        property_dict[name] = property_dict_i
    return property_dict


def temperature_vector(T):
    T0 = 298.15  # K
    res = np.array([1,
                    1/T - 1/T0,
                    np.log(T/T0),
                    T - T0,
                    T**2 - T0**2,
                    1/T**2 - 1/T0**2], dtype=np.double)
    return res


def make_parameter_matrix(species, parameter, property_dict):
    indexes = []
    values = []

    for i, specie1 in enumerate(species):
        for j, specie2 in enumerate(species):
            key = tuple(sorted([specie1, specie2]))
            if key in property_dict[parameter]:
                res_ij = np.array(property_dict[parameter][key])
                if (i == j):
                    res_ij *= 2
                values.append(res_ij)
                indexes.append((i, j))
    M = np.array(values, dtype=np.double)
    M_inds = np.array(indexes, dtype=np.intc)
    if M.shape[0] == 0: #Case of an empty matrix
        M = M.reshape(0, 6)
        M_inds = M_inds.reshape(0, 2)
    return M, M_inds


def make_parameter_3_tensor(species, parameter, property_dict):
    indexes = []
    values = []

    for i, specie1 in enumerate(species):
        for j, specie2 in enumerate(species):
            for k, specie3 in enumerate(species):
                key = tuple(sorted([specie1, specie2, specie3]))
                if key in property_dict[parameter]:
                    res_ij = np.array(property_dict[parameter][key])
                    if (i == j) and (i == k):
                        res_ij *= 3
                    elif (i == j) or (i == k) or (j == k):
                        res_ij *= 2
                    values.append(res_ij)
                    indexes.append((i, j, k))
    M = np.array(values, dtype=np.double)
    M_inds = np.array(indexes, dtype=np.intc)
    if M.shape[0] == 0: #Case of an empty matrix
        M = M.reshape(0, 6)
        M_inds = M_inds.reshape(0, 3)
    return M, M_inds


def _find_and_replace_charge_signed(string, sign):
    # Cation finding
    pattern = r'.*(\%s\d).*'%sign
    match = re.search(pattern, string)
    if match:
        number = int(match.group(1)[-1])
        patternsub = r'\%s\d' % sign
        new_string = re.sub(patternsub, sign*number, string)
    else:
        new_string = string
    return new_string


def _find_and_replace_charge(string):
    string = _find_and_replace_charge_signed(string, '+')
    string = _find_and_replace_charge_signed(string, '-')
    return string


def _remove_after_hash(linestrings):
    for i, string in enumerate(linestrings):
        if '#' in string:
            return linestrings[:i]
    return linestrings


def _process_line_pitzer(line):
    linestrings = line.split()
    if len(linestrings) == 1:  # Parameter name
        return linestrings
    linestrings = _remove_after_hash(linestrings)  # Remove comments
    for i, string in enumerate(linestrings):
        try:  # Should be a float
            linestrings[i] = float(string)
        except:  # Is a element
            linestrings[i] = _find_and_replace_charge(string)
    max_size = 8 if type(linestrings[2]) == float else 9
    if len(linestrings) < max_size:
        linestrings = linestrings + [0.0]*(max_size - len(linestrings))
    return linestrings
