""" Functions used by GE for evolving neural networks"""
# -*- coding: utf-8 -*-

import numpy as np
import math

import warnings
warnings.filterwarnings('ignore', message='overflow encountered in exp')
    
def activate(a: np.ndarray) -> np.ndarray:
    """ Sigmoid activation functioning matches behavior of original ATHENA code
    
    Args:
        a: contains values to modify
    
    Returns:
        np.ndarray containing modified values
    
    """

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

def PA(inputs:list) -> np.ndarray:
    """ Additive node 
    
    Args:
        inputs: np.ndarrays where each value is added
    
    Returns:
        np.ndarray containing the result modified by sigmoid function    
    """

    return activate(sum(inputs))
    
def PM(inputs:list) -> np.ndarray:
    """ Multiplicative node 
    
    Args:
        inputs: np.ndarrays where each value is multiplied
    
    Returns:
        np.ndarray containing the result modified by sigmoid function    
    """

    result = inputs[0]
    for val in inputs[1:]:
        result = result * val
    return activate(result)
    
def PS(inputs: list) -> np.ndarray:
    """ Subtraction node

    Args:
        inputs: np.ndarrays where first is minuend and all the rest are
            subtrahends
    
    Returns:
        np.ndarray containing the difference modified by sigmoid function    
    """

    diff = inputs[0]
    for num in inputs[1:]:
        diff = diff - num
    return activate(diff)
    
def PD(inputs:list) -> np.ndarray:
    """ function to match ATHENA handling of division node in network. Uses 
     protected dividision

    Args:
        inputs: np.ndarrays where first is numerator and all the rest are
            denominators
    
    Returns:
        np.ndarray containing the dividend modified by sigmoid function

    """

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
    return activate(result)


def add(a:np.ndarray,b:np.ndarray) -> np.ndarray:
    """ addition operator """
    return a+b

def sub(a:np.ndarray,b:np.ndarray) -> np.ndarray:
    """ subtraction operator """
    return a-b

def mult(a:np.ndarray,b:np.ndarray) -> np.ndarray:
    """ multiplication operator """
    return a*b

def div(a:np.ndarray,b:np.ndarray) -> np.ndarray:
    """ division operator """
    return a/b if b != 0 else 1.0         

def pdiv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Koza's protected division modified to work with numpy arrays as used in PonyGE2.
    Returns 1 when denominator is zero

    Args:
        a: numerator
        b: denominator
    
    Returns:
        numpy array with a/b or 1 when denominator is 0
    """

    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b == 0, np.ones_like(a), a / b)
    except ZeroDivisionError:
        # In this case we are trying to divide two constants, one of which is 0
        # Return a constant.
        return 1.0

def PAND(inputs: list) -> np.ndarray:
    """
    Boolean AND operator. Returns 1.0 if all inputs are > 0 and 0 otherwise.
    Ignores np.nan inputs

    Args:
        inputs: np.ndarrays containing floats or np.nan
    
    Returns:
        numpy array 
    """
    stacked_arrays = np.stack(inputs, axis=0)
    
    valid_mask = ~np.isnan(stacked_arrays)
    all_nan_mask = np.all(np.isnan(stacked_arrays), axis=0)
    
    result = np.all((stacked_arrays > 0) | ~valid_mask, axis=0)    
    result = np.where(all_nan_mask, np.nan, result)
    
    return result.astype(float)


def PNAND(inputs: list) -> np.ndarray:
    """
    Boolean NAND operator. Returns 0.0 if all inputs are > 0 and 1 otherwise.
    Ignores np.nan inputs

    Args:
        inputs: np.ndarrays containing floats or np.nan
    
    Returns:
        numpy array 
    """
    stacked_arrays = np.stack(inputs, axis=0)
    
    valid_mask = ~np.isnan(stacked_arrays)    
    all_nan_mask = np.all(np.isnan(stacked_arrays), axis=0)
    result = np.all((stacked_arrays <= 0) | ~valid_mask, axis=0)    
    result = np.where(all_nan_mask, np.nan, result)
    
    return result.astype(float)

def POR(inputs: list) -> np.ndarray:
    """
    Boolean OR operator. Returns 1.0 if any inputs are > 0 and 1 otherwise.
    Ignores np.nan inputs

    Args:
        inputs: np.ndarrays containing floats or np.nan
    
    Returns:
        numpy array 
    """
    stacked_arrays = np.stack(inputs, axis=0)
    
    valid_mask = ~np.isnan(stacked_arrays)    
    all_nan_mask = np.all(np.isnan(stacked_arrays), axis=0)
    
    result = np.any((stacked_arrays > 0) & valid_mask, axis=0)  
    result = np.where(all_nan_mask, np.nan, result)
    
    return result.astype(float)


def PNOR(inputs: list) -> np.ndarray:
    """
    Boolean NOR operator. Returns 0.0 if any inputs are > 0 and 1.0 otherwise.
    Ignores np.nan inputs

    Args:
        inputs: np.ndarrays containing floats or np.nan
    
    Returns:
        numpy array 
    """
    stacked_arrays = np.stack(inputs, axis=0)
    
    valid_mask = ~np.isnan(stacked_arrays)    
    all_nan_mask = np.all(np.isnan(stacked_arrays), axis=0)
    
    any_positive_mask = np.any((stacked_arrays > 0) & valid_mask, axis=0)    
    all_non_positive_mask = np.all((stacked_arrays <= 0) | np.isnan(stacked_arrays), axis=0)
    
    result = np.full(stacked_arrays.shape[1:], np.nan, dtype=float)
    result[any_positive_mask] = 0.0    
    result[all_non_positive_mask] = 1.0
    
    final_result = np.where(all_nan_mask, np.nan, result)
    
    return final_result


def PXOR(inputs: list) -> np.ndarray:
    """
    Boolean XOR operator. Returns 1.0 if if inputs alternate >0 and <=0.
    Returns 1.0 if they do not.
    Ignores np.nan inputs

    Args:
        inputs: np.ndarrays containing floats or np.nan
    
    Returns:
        numpy array 
    """
    stacked_arrays = np.stack(inputs, axis=0)
    valid_mask = ~np.isnan(stacked_arrays)
    all_nan_mask = np.all(np.isnan(stacked_arrays), axis=0)
    
    result = np.zeros(stacked_arrays.shape[1:], dtype=float)
    
    # Iterate over each position and check for alternating behavior
    for idx in np.ndindex(stacked_arrays.shape[1:]):
        # Get the values for this index across all arrays
        values = stacked_arrays[:, idx]
        
        # Ignore NaN values and get the valid (non-NaN) values
        valid_values = values[valid_mask[:, idx]]
        
        if len(valid_values) > 1:
            # Check if the values alternate between > 0 and <= 0
            # We want a sequence of values like: [>0, <=0, >0, <=0] or the reverse
            alternating = True
            for i in range(1, len(valid_values)):
                if valid_values[i] > 0 and valid_values[i-1] > 0:
                    alternating = False
                    break
                if valid_values[i] <= 0 and valid_values[i-1] <= 0:
                    alternating = False
                    break
            
            if alternating:
                result[idx] = 1.0
        elif len(valid_values) == 1:
            # If only one valid value exists and it's > 0, set result to 0
            result[idx] = 0.0
    
    # Apply np.nan where all values were np.nan at the position
    result = np.where(all_nan_mask, np.nan, result)
    
    return result
