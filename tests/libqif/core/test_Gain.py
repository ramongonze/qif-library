from libqif.core.secrets import Secrets
from libqif.core.channel import Channel
from libqif.core.hyper import Hyper
from libqif.core.gvulnerability import Gain
import numpy as np
import unittest
import os

class TestGain(unittest.TestCase):
    def setUp(self):
        self.epsilon = 10**(-6)
        self.secrets1 = Secrets(['x1','x2'], [0.3,0.7])
        self.channel1 = Channel(self.secrets1, ['y1','y2'], np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ]))
        self.gain1 = Gain(self.secrets1, ['w1','w2','w3','w4','w5'], np.array([
            [-1.0,  1.0],
            [ 0.0,  0.5],
            [ 0.4,  0.1],
            [ 0.8, -0.9],
            [ 0.1,  0.2]
        ]))

        # Example 5.1 from the book The Science of Quantitative Information Flow
        self.secrets2 = Secrets(['x1','x2','x3','x4','x5'], [9/10,1/40,1/40,1/40,1/40])
        self.channel2 = Channel(self.secrets2, ['y1','y2'], np.array([
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]
        ]))
        self.gain2 = Gain(self.secrets2, ['w1','w2','w3','w4','w5'], np.identity(5))

        self.secrets3 = Secrets(['x1','x2','x3'], [1/4, 1/2, 1/4])
        self.channel3 = Channel(self.secrets3, ['y1','y2','y3','y4'], np.array([
            [1/2, 1/2,  0,    0],
            [  0, 1/4, 1/2, 1/4],
            [1/2, 1/3, 1/6,   0]
        ]))
        self.gain3 = Gain(self.secrets3, ['w1','w2','w3'], np.identity(3))

        # Example 5.16 from the book The Science of Quantitative Information Flow
        self.secrets4 = Secrets(['x1','x2'], [1/100,99/100])
        self.channel4 = Channel(self.secrets4, ['positive','negative'], np.array([
            [9/10, 1/10],
            [1/10, 9/10]
        ]))
        self.gain4 = Gain(self.secrets4, ['w1','w2'], np.identity(2))

    def test_valid_gains(self):
        self.assertLess(self.gain1.prior_vulnerability() - 0.4, self.epsilon)
        self.assertLess(self.gain1.posterior_vulnerability(Hyper(self.channel1)) - 0.5575, self.epsilon)
        self.assertLess(self.gain2.prior_vulnerability() - 9/10, self.epsilon)
        self.assertLess(self.gain3.prior_vulnerability() - 1/2, self.epsilon)
        self.assertLess(self.gain3.posterior_vulnerability(Hyper(self.channel3)) - 5/8, self.epsilon)
        # self.assertLess(self.gain4.prior_vulnerability() - 99/100, self.epsilon)
        # self.assertLess(self.gain4.posterior_vulnerability(Hyper(self.channel4)), self.epsilon)
