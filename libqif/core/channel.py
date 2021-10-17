"""QIF channels."""

from util.probability import check_prob_distribution
from util.types import check_list, check_numpy_array
from numpy import arange

class Channel:

    def __init__(self, secrets, outputs, channel):
        """QIF channel. 

        Parameters
        ----------
        secrets : core.Secrets
            Set of secrets.

        outputs : list
            Outputs labels.

        channel : numpy.ndarray
            Channel matrix. Each line must be a probability distribution.
        """

        self.secrets = secrets
        self.outputs_labels = check_list(outputs)
        self.num_outputs = len(outputs)
        self.matrix = self._check_channel_matrix(channel)

    def _check_channel_matrix(self, channel):
        check_numpy_array(channel)
        if (self.secrets.num_secrets != channel.shape[0] or 
            self.num_outputs != channel.shape[1]):
                raise Exception('The matrix shape does not match with the ' +
                                'set of secrets or with the set of outputs')
        
        # Check if each line of the channel matrix is a probability distribution
        for i in arange(channel.shape[0]):
            check_prob_distribution(channel[i])

        return channel
