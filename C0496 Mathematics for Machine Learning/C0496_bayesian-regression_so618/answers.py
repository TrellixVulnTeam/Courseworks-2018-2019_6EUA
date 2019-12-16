# -*- coding: utf-8 -*-

"""
Use this file for your answers.

This file should been in the root of the repository
(do not move it or change the file name)

"""

import numpy as np

def lml(alpha, beta, Phi, Y):
    """
    4 marks

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: the log marginal likelihood, a scalar
    """
    N = len(Y)
    phi_phi = Phi.dot(Phi.T)
    S_N = alpha * phi_phi + beta * np.identity(N)
    first_part = - N * 0.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(S_N))
    last_part = - 0.5 * (Y.T.dot(np.linalg.inv(S_N))).dot(Y)
    return (first_part + last_part)[0][0]


def grad_lml(alpha, beta, Phi, Y):
    """
    8 marks (4 for each component)

    :param alpha: float
    :param beta: float
    :param Phi: array of shape (N, M)
    :param Y: array of shape (N, 1)
    :return: array of shape (2,). The components of this array are the gradients
    (d_lml_d_alpha, d_lml_d_beta), the gradients of lml with respect to alpha and beta respectively.
    """
    N = len(Y)
    phi_phi = Phi.dot(Phi.T)
    S_N = alpha * phi_phi + beta * np.identity(N)
    det_SN = np.linalg.det(S_N)
    inv_SN = np.linalg.inv(S_N)
    first_part_a = - 0.5 * np.matrix.trace(inv_SN.dot(phi_phi))
    last_part_a = 0.5 * (((Y.T.dot(inv_SN)).dot(phi_phi)).dot(inv_SN)).dot(Y)
    first_part_b = - 0.5 * np.matrix.trace(inv_SN)
    last_part_b = 0.5 * Y.T.dot(np.dot(inv_SN, inv_SN)).dot(Y)
    grad_a = (first_part_a + last_part_a)[0][0]
    grad_b = (first_part_b + last_part_b)[0][0]
    return np.reshape(np.array([grad_a, grad_b]), (2, ))
