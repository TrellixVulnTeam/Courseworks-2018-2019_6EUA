# -*- coding: utf-8 -*-

"""
Use this file for your answers. 

This file should been in the root of the repository 
(do not move it or change the file name) 

"""

# NB this is tested on python 2.7. Watch out for integer division

import numpy as np
    
def grad_f1(x):
    """
    4 marks

    :param x: input array with shape (2, )
    :return: the gradient of f1, with shape (2, )
    """
    return np.array([8 * x[0] - 2 * x[1] - 1, 8 * x[1] - 2 * x[0] - 1])

def grad_f2(x):
    """
    6 marks

    :param x: input array with shape (2, )
    :return: the gradient of f2, with shape (2, )
    """
    return np.array(
            [np.cos(x[0]**2 - 2*x[0] + 1 + x[1]**2)*(2*x[0] - 2) + 6*x[0] - 2*x[1] - 2, 
             np.cos(x[0]**2 - 2*x[0] + 1 + x[1]**2)*2*x[1] + 6*x[1] - 2*x[0] + 6]
            )

def grad_f3(x):
    """
    This question is optional. The test will still run (so you can see if you are correct by
    looking at the testResults.txt file), but the marks are for grad_f1 and grad_f2 only.

    Do not delete this function.

    :param x: input array with shape (2, )
    :return: the gradient of f3, with shape (2, )
    """
    A = - 3*x[0]**2 - 3*x[1]**2 + 2*x[0]*x[1] + 2*x[0] - 6*x[1] - 3
    E = - (x[0] - 1)**2 - x[1]**2
    D = x[0]**2 + x[1]**2 + 1./100
    return np.array(
            [- np.exp(E)*(-2*x[0] + 2) - np.exp(A)*(-6*x[0] + 2*x[1] + 2) + 1./5 * x[0] / D, 
             - np.exp(E)*(-2*x[1]) - np.exp(A)*(-6*x[1] + 2*x[0] - 6) + 1./5 * x[1] / D]
            )


