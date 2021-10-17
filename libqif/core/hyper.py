"""Hyper-distributions."""

from numpy import array, arange, zeros

class Hyper:

    def __init__(self, channel):
        """Hyper-distribution.

        Parameters
        ----------
        channel : core.Channel
            Channel.
        """

        self.channel = channel
        self.joint = self._generate_joint_distribution()
        self.outer, self.inners = self._generate_posteriors()

    def _generate_joint_distribution(self):
        joint = []
        channel_t = self.channel.matrix.T
        for i in arange(self.channel.num_outputs):
            joint.append(self.channel.secrets.prior * channel_t[i])

        return array(joint).T            

    def _generate_posteriors(self):
        joint_t = self.joint.T.copy()
        outer = []
        for i in arange(self.channel.num_outputs):
            outer.append(joint_t[i].sum())
            if outer[i] > 0:
                joint_t[i] = joint_t[i]/outer[i]
        
        return array(outer), joint_t.T
