"""Set of secrets."""

from util.types import check_int, check_list
from util.probability import check_prob_distribution
from numpy import array

class Secrets:
    
    def __init__(self, num_secrets, labels, prob):
        """Set of secrets.

        Parameters
        ----------
        num_secrets : int
            Number of secrets.

        labels : list
            Secrets labels.

        prob : list, numpy.ndarray
            Array of probability distributions. prob[i] is the probability of
            secret with i-th label happen.
        """
        self.num_secrets = check_int(num_secrets)
        self.labels = self._check_labels(labels)
        self.prob = array(check_prob_distribution(prob))

    def _check_labels(self, labels):
        """Check if the parameter is a list and has |labels| = number of secrets."""
        check_list(labels)
        if len(labels) != self.num_secrets:
            raise Exception('The number of labels is different from the number of secrets')
        return labels        
