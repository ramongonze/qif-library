from libqif.core.secrets import Secrets
from libqif.core.channel import Channel
from libqif.core.hyper import Hyper
from libqif.core.luncertainty import LUncertainty
import numpy as np
import unittest
import os

class TestLUncertainty(unittest.TestCase):
    def setUp(self):
        self.epsilon = 10**(-6)
        self.secrets1 = Secrets(['x1','x2'], [0.3,0.7])
        self.channel1 = Channel(self.secrets1, ['y1','y2'], np.array([
            [0.75, 0.25],
            [0.25, 0.75]
        ]))
        self.loss1 = LUncertainty(self.secrets1, ['w1','w2','w3','w4','w5'], [
            [-1.0,  1.0],
            [ 0.0,  0.5],
            [ 0.4,  0.1],
            [ 0.8, -0.9],
            [ 0.1,  0.2]
        ])

        # Example 5.1 from the book The Science of Quantitative Information Flow
        self.secrets2 = Secrets(['x1','x2','x3','x4','x5'], [9/10,1/40,1/40,1/40,1/40])
        self.channel2 = Channel(self.secrets2, ['y1','y2'], np.array([
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]
        ]))
        self.loss2 = LUncertainty(self.secrets2, ['w1','w2','w3','w4','w5'], np.identity(5))

        self.secrets3 = Secrets(['x1','x2','x3'], [1/4, 1/2, 1/4])
        self.channel3 = Channel(self.secrets3, ['y1','y2','y3','y4'], np.array([
            [1/2, 1/2,  0,    0],
            [  0, 1/4, 1/2, 1/4],
            [1/2, 1/3, 1/6,   0]
        ]))
        self.loss3 = LUncertainty(self.secrets3, ['w1','w2','w3'], np.identity(3))

        # Example 5.16 from the book The Science of Quantitative Information Flow
        self.secrets4 = Secrets(['x1','x2'], [1/100,99/100])
        self.channel4 = Channel(self.secrets4, ['positive','negative'], np.array([
            [9/10, 1/10],
            [1/10, 9/10]
        ]))
        self.loss4 = LUncertainty(self.secrets4, ['w1','w2'], np.identity(2))

    def test_valid_losses(self):
        self.assertLess(self.loss1.prior_uncertainty() + 0.39, self.epsilon)
        self.assertLess(self.loss1.posterior_uncertainty(Hyper(self.channel1)) + 0.4624999999, self.epsilon)
        self.assertLess(self.loss2.prior_uncertainty() - 1/40, self.epsilon)
        self.assertLess(self.loss2.posterior_uncertainty(Hyper(self.channel2)), self.epsilon)
        self.assertLess(self.loss3.prior_uncertainty() - 1/4, self.epsilon)
        self.assertLess(self.loss3.posterior_uncertainty(Hyper(self.channel3)) - 0.0833333333, self.epsilon)
        self.assertLess(self.loss4.prior_uncertainty() - 1/100, self.epsilon)
        self.assertLess(self.loss4.posterior_uncertainty(Hyper(self.channel4)) - 1/100, self.epsilon)

        loss = LUncertainty(self.secrets1, ['w1','w2'], lambda w,x : 1 if w == x else 0)
        np.testing.assert_array_equal(loss.matrix, np.identity(2))

    def test_invalid_losses(self):
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, [], lambda x : x)
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, [], np.identity(2))
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, ['w1','w2'], [[],[]])
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, ['w1','w2'], np.array([]))
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, ['w1','w2'], [])
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, ['w1','w2'], lambda x : x)
        with self.assertRaises(Exception):
            LUncertainty(self.secrets1, ['w1','w2'], lambda x : 1)
        with self.assertRaises(Exception):
            LUncertainty(1, ['w1','w2'], np.identity(2))

    def test_leakage(self):
        add_leakage, mult_leakage = self.loss1.leakage(Hyper(self.channel1))
        self.assertLess(add_leakage - (0.4624999999 + 0.39), self.epsilon)
        self.assertLess(mult_leakage - (0.4624999999/0.39), self.epsilon)

        add_leakage, mult_leakage = self.loss2.leakage(Hyper(self.channel2))
        self.assertLess(add_leakage - (- 1/40), self.epsilon)
        self.assertLess(mult_leakage - (0), self.epsilon)
        
        add_leakage, mult_leakage = self.loss3.leakage(Hyper(self.channel3))
        self.assertLess(add_leakage - (0.0833333333 - 1/4), self.epsilon)
        self.assertLess(mult_leakage - (0.0833333333/(1/4)), self.epsilon)

        add_leakage, mult_leakage = self.loss4.leakage(Hyper(self.channel4))
        self.assertLess(add_leakage, self.epsilon)
        self.assertLess(mult_leakage - 1, self.epsilon)
        