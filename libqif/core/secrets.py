"""Set of secrets."""

from util.types import check_list
from util.probability import check_prob_distribution
from numpy import array

class Secrets:
    
    def __init__(self, secrets, prior):
        """Set of secrets.

        Parameters
        ----------
        secrets : list
            Secrets labels.

        prior : list, numpy.ndarray
            Prior distribution on the set of secrets. prior[i] is the
            probability of secret with i-th label beeing the real secret.
        """
        self.labels = check_list(secrets)
        self.num_secrets = len(secrets)
        check_prob_distribution(prior)
        self.prior = array(prior)
