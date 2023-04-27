"""l-uncertainty"""

from libqif.util.types import is_list, is_function, is_2d_list_matrix, is_2d_numpy_matrix
from libqif.core.secrets import Secrets
from numpy import arange, zeros, array
from numpy import min as npmin

class LUncertainty:

    def __init__(self, secrets, actions, lfunction):
        """:math:`\ell`-uncertainty. To create an instance of this class it is necessary 
        to have an instance of :py:class:`.Secrets` class and a loss function, 
        that can be a matrix or a pointer to a function. The matrix G must be 
        G :math:`w{\\times}n` where :math:`w` is the number of actions, :math:`n` is the 
        number of secrets and :code:`G[w][x]` is the adversary's loss when she takes the action 
        :code:`w` and the secret's value is :code:`x`. The function must have 
        two input parameters (action w and secret x, in this order) and outputs a real value.

        Attributes
        ----------
        secrets : core.Secrets
            Secrets object.

        actions : list
            List of actions' labels.

        num_actions : int
            Number of actions.

        matrix : numpy.ndarray
            Loss function matrix. :code:`loss[w][x]` is the adversary's loss
            when she takes the action of index :code:`w` (that has the
            label :code:`actions[w]`) and the secret is the one from index
            :code:`x` (and has the label :code:`secrets.labels[x]`).

        Parameters
        ----------
        secrets : core.Secrets
            Set of secrets.

        actions : list
            Set of actions.

        lfunction : list, numpy.ndarray, pointer to a function
            A 2d matrix or a pointer to a loss function.
            If the value is a matrix, its shape must match with the actions
            and secrets sets size.
            If the value is a pointer to a function, the function must have
            2 input parameters (w,x), where w is the index of an element from
            the set of actions and x is an index of an element from the set
            of secrets.
        """

        self._check_types(secrets, actions, lfunction)
        self._check_sizes(secrets, actions, lfunction)
        self.secrets = secrets
        self.actions = actions
        self.num_actions = len(self.actions)
        self.matrix = None
        self._set_loss_function_matrix(lfunction)        

    def prior_uncertainty(self):
        """Prior uncertainty.
        
        Returns
        -------
        prior_uncertainty : float
            Prior uncertainty.
        """
        return npmin(self.secrets.prior @ self.matrix.T)
    
    def posterior_uncertainty(self, hyper):
        """Posterior uncertainty.

        Parameters
        ----------
        hyper : core.Hyper
            Hyper-distribution.

        Returns
        -------
        posterior_uncertainty : float
            Posterior uncertainty.
        """

        if hyper.channel.secrets.num_secrets != self.secrets.num_secrets:
            raise Exception('The number of secrets in the loss function is' +
                            'different from the one in the hyper-distribution.')

        return npmin(self.matrix @ hyper.joint, axis=0).sum()

    def leakage(self, hyper):
        """Calculates the additive and multiplicative leakages.

        Parameters
        ----------
        hyper : core.hyper.Hyper
            Hyper-distribution

        Returns
        -------
        add_leakage, mult_leakage : (float, float)
            Additive and multiplicative leakage
        """

        prior_v = self.prior_uncertainty()
        posterior_v = self.posterior_uncertainty(hyper)

        add_leakage = posterior_v - prior_v
        
        if prior_v == 0:
            mult_leakage = 0
        else:
            mult_leakage = posterior_v/prior_v

        return add_leakage, mult_leakage

    def _check_types(self, secrets, actions, lfunction):
        if type(secrets) != type(Secrets(['x1','x2'], [1,0])):
            raise TypeError('The parameter \'secrets\' must be a core.Secrets object')
        
        if not is_list(actions) and not is_numpy_array(actions):
            raise TypeError('The parameter \'actions\' must be a list or a numpy.ndarray')

        if not is_2d_list_matrix(lfunction) and not is_2d_numpy_matrix(lfunction) and not is_function(lfunction):
            raise TypeError('The parameter \'lfunction\' must be a 2d matrix or a pointer to a function')

    def _check_sizes(self, secrets, actions, lfunction):
        if not is_function(lfunction) and len(actions) != len(lfunction):
            raise Exception('The number of rows in the loss function matrix ' +
                            'must have the same number of actions in the ' +
                            'labels list')

        if len(actions) < 1:
            raise Exception('The set of actions must have at least one element')
        
        if not is_function(lfunction):
            for i in arange(len(actions)):
                if len(lfunction[i]) < 0:
                    raise Exception('There is an empty row in the loss matrix')

        if not is_function(lfunction) and secrets.num_secrets != len(lfunction[0]):
            raise Exception('The number of columns in the loss function matrix ' +
                            'must be the same as the number of secrets')

    def _build_loss_matrix(self, lfunction):
        """Given a loss function build the matrix G."""
        self.matrix = zeros((self.num_actions, self.secrets.num_secrets))
        
        for w in arange(self.num_actions):
            for x in arange(self.secrets.num_secrets):
                self.matrix[w][x] = lfunction(w,x)
    
    def _check_loss_matrix(self, matrix):
        """Check if a loss function matrix is valid."""
        if (self.num_actions != len(matrix) or
            self.secrets.num_secrets != len(matrix[0])):
            raise Exception('Loss function matrix shape does not match with ' +
                            'the set of secrets or the set of actions size.')

    def _set_loss_function_matrix(self, lfunction):
        try:
            if is_function(lfunction):
                # Generate the loss matrix for all actions and secrets
                self._build_loss_matrix(lfunction)
            else:
                # Copy the loss matrix
                self._check_loss_matrix(lfunction)
                self.matrix = array(lfunction).copy()
        except:
            raise Exception(
                'Invalid loss function. It must be a pointer to a function ' +
                'that has 2 input parameters (action\'s index and ' +
                'secret\'s index) or a 2d matrix.'
            )
