"""Set of secrets."""

from libqif.util.types import check_list
from libqif.util.probability import check_prob_distribution
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
        self._check_labels_and_prior(secrets, prior)
        self.labels = check_list(secrets)
        self.num_secrets = len(secrets)
        check_prob_distribution(prior)
        self.prior = array(prior)

    def _check_labels_and_prior(self, secrets, prior):
        """Check if the size of the list of labels is the same of the prior
        distribution.
        """
        if len(secrets) != len(prior):
            raise Exception('The size of label\'s list is different from ' +
                            'the number of elements in the prior distribution')
