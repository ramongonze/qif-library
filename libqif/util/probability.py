"""Util methods related to probability distributions."""

from numpy import arange

def check_prob_distribution(prob):
    """Chech wheter an array is a probability distribution or not.
    All the values must be in the interval [0,1] and they must sum up to 1.
    It raises an exception if the array is not a probability distribution or
    does nothing if it is a valid probability distribution.

    Parameters
    ----------
    prob : list, numpy.ndarray
        Array containing a probability distribution.
    """

    epsilon = 0.000001 # Used to compare probability distributions
    for i in arange(len(prob)):
        if prob[i] < 0 or prob[i] > 1:
            raise ValueError('The values must be in the interval [0,1]')
    
    prob_sum = sum(prob)
    if prob_sum < 1-epsilon or prob_sum > 1+epsilon:
        raise ValueError('All the values must sum up to 1 (with an error of ' +
                         'at most 10^(-6)')
