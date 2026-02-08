"""
Input validation utilities.
"""

import numpy as np


def check_input(X):
    """
    Validate input data.
    
    Parameters
    ----------
    X : array-like
        Input data
        
    Returns
    -------
    X : ndarray
        Validated and converted input
    """
    # TODO: Implement input validation
    # Use sklearn.utils.check_array when implementing
    return np.asarray(X)
