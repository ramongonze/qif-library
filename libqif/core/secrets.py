"""Set of secrets."""

from util.types import check_list
from util.probability import check_prob_distribution
from numpy import array

class Secrets:
    
    def __init__(self, secrets, prob):
        """Set of secrets.

        Parameters
        ----------
        secrets : set
            Secrets labels.

        prob : list, numpy.ndarray
            Array of probability distributions. prob[i] is the probability of
            secret with i-th label happen.
        """
        self.labels = check_list(secrets)
        self.num_secrets = len(secrets)
        self.prob = array(check_prob_distribution(prob))
