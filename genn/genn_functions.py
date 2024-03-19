#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

import warnings
warnings.filterwarnings('ignore', message='overflow encountered in exp')

# def activate_sigmoid(a):
#     if isinstance(a, np.ndarray):
#         exceeded = (a < -709) | (a > 709)
#         a[exceeded] = np.where(a[exceeded] > 709, np.ones_like(a[exceeded]), np.zeros_like(a[exceeded]))
#         notexceeded = np.invert(exceeded)
#         a[notexceeded] = 1.0 / (1.0 + np.exp(np.negative(a[notexceeded])))
#     else:
#         if a >= -709 and a <= 709:
#             a = 1.0 / (1.0 + np.exp(-a))
#         else:
#             a = np.float64(0.0) if a < -709 else np.float64(1.0)
#     
#     return a
    
# simplified version better matching ATHENA sigmoid function
def activate_sigmoid(a):
    if isinstance(a, np.ndarray):
        a = np.where(a <= -709, -708, a)
        a = np.where(a >= 709, 708, a)
#         a = 1.0 / (1.0 + np.exp(np.negative(a)))
        a = 1.0 / (1.0 + np.exp(-a))
    else:
        if a >= -709 and a <= 709:
            a = 1.0 / (1.0 + np.exp(-a))
        else:
            a = np.float64(0.0) if a < -709 else np.float64(1.0)
    
    return a


#     if val < 709 and val > -709:    
#         return 1.0 / (1.0 + math.exp(-val))
    
def PA(inputs):
    return activate_sigmoid(sum(inputs))
    
def PM(inputs):
    result = inputs[0]
    for val in inputs[1:]:
        result = result * val
    return activate_sigmoid(result)
    
def PS(inputs):
    diff = inputs[0]
    for num in inputs[1:]:
        diff = diff - num
    return activate_sigmoid(diff)

# def PD(inputs):
#     result = inputs[0]
#     for divisor in inputs[1:]:
#         try:
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 result = np.where(divisor == 0, np.ones_like(divisor), result / divisor)        
#         except ZeroDivisionError:
#             return 1.0
#     return activate_sigmoid(result)
# 
#     quotient = inputs[0]
#     for num in inputs[1:]:
#         if num == 0:
#             return 1
#         quotient /= num
#     return activate_sigmoid(quotient)
    
# updated PDiv function to match ATHENA
def PD(inputs):
    result = inputs[0]
    anyzeroed = np.full_like(result,False).astype(bool)
    for divisor in inputs[1:]:
        try:
            zeroed = (divisor == 0)
            anyzeroed[zeroed]=True
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(divisor == 0, np.ones_like(result), result / divisor)
#                 result = np.where(divisor == 0, np.ones_like(divisor), result / divisor)
                        
        except ZeroDivisionError:
            return 1.0
    
    result[anyzeroed]=1000
    return activate_sigmoid(result)


def add(a,b):
    return a+b

def sub(a,b):
    return a-b

def mult(a,b):
    return a*b

def div(a,b):
     return a/b if b != 0 else 1.0
#     with np.errstate(divide='ignore', invalid='ignore'):
#         return np.where(b == 0, np.ones_like(b), a / b)           

def pdiv(x, y):
    """
    Koza's protected division is:

    if y == 0:
      return 1
    else:
      return x / y

    but we want an eval-able expression. The following is eval-able:

    return 1 if y == 0 else x / y

    but if x and y are Numpy arrays, this creates a new Boolean
    array with value (y == 0). if doesn't work on a Boolean array.

    The equivalent for Numpy is a where statement, as below. However
    this always evaluates x / y before running np.where, so that
    will raise a 'divide' error (in Numpy's terminology), which we
    ignore using a context manager.

    In some instances, Numpy can raise a FloatingPointError. These are
    ignored with 'invalid = ignore'.

    :param x: numerator np.array
    :param y: denominator np.array
    :return: np.array of x / y, or 1 where y is 0.
    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(y == 0, np.ones_like(x), x / y)
    except ZeroDivisionError:
        # In this case we are trying to divide two constants, one of which is 0
        # Return a constant.
        return 1.0

