"""QIF channels."""

from libqif.core.secrets import Secrets
from libqif.util.probability import check_prob_distribution
from libqif.util.types import is_list, is_2d_list_matrix, is_2d_numpy_matrix
from numpy import arange, array

class Channel:

    def __init__(self, secrets, outputs, channel):
        """QIF channel. 

        Attributes
        ----------
        secrets : core.Secrets
            Set of secrets.

        outputs_labels : list
            Channel outputs labels.

        num_ouputs : int
            Number of channel's outputs.

        matrix : numpy.ndarray
            Channel matrix.

        Parameters
        ----------
        secrets : core.Secrets
            Set of secrets.

        outputs : list
            Outputs labels.

        channel : numpy.ndarray
            Channel matrix. Each line must be a probability distribution.
        """
        self._check_types(secrets, outputs, channel)
        self._check_sizes(secrets, outputs, channel)
        self._check_channel_matrix(channel)
        self.secrets = secrets
        self.outputs_labels = outputs
        self.matrix = array(channel)
        self.num_outputs = len(outputs)

    def _check_types(self, secrets, outputs, channel):
        if type(secrets) != type(Secrets(['x1','x2'], [1,0])):
            raise TypeError('The parameter \'secrets\' must be a core.secrets.Secrets object')

        if not is_list(outputs):
            raise TypeError('The parameter \'outputs\' must be a list')

        if not is_2d_list_matrix(channel) and not is_2d_numpy_matrix(channel):
            raise TypeError('The parameter \'channel\' must be a 2d matrix ' +
                            '(list of lists or a numpy.ndarray with 2 dimensions)')
    
    def _check_sizes(self, secrets, outputs, channel):
        if secrets.num_secrets != channel.shape[0]:
            raise Exception('The number of rows in channel matrix must be the ' +
                            'same as the number of secrets') 

        if len(outputs) != channel.shape[1]:
            raise Exception('The number of columns in channel matrix must be ' +
                            'the same as the number of outputs (second parameter)')

    def _check_channel_matrix(self, channel):        
        # Check if each line of the channel matrix is a probability distribution
        for i in arange(channel.shape[0]):
            check_prob_distribution(channel[i])
