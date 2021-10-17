"""Set of secrets."""

from util.types import check_list
from util.probability import check_prob_distribution
from numpy import array

class Secrets:
    
    def __init__(self, secrets, prior):
        """Set of secrets.

        Attributes
        ----------
        labels : list
            List of secret's labels.

        num_secrets : int
            Number of secrets.

        prior : numpy.ndarray
            Prior distribution on the set of secrets. prior[i] is the
            probability of secret named labels[i] beeing the real secret.


        Parameters
        ----------
        secrets : list
            Secrets labels.

        prior : list, numpy.ndarray
            Prior distribution on the set of secrets. prior[i] is the
            probability of secret named labels[i] beeing the real secret.
        """
        self.labels = check_list(secrets)
        self.num_secrets = len(secrets)
        check_prob_distribution(prior)
        self.prior = array(prior)
