from libqif.core.secrets import Secrets
from libqif.core.channel import Channel
import numpy as np
import unittest
import os

class TestChannel(unittest.TestCase):
    def setUp(self):
        self.secrets1 = Secrets(['x1','x2','x3'], [1/3,1/3,1/3])
        self.channel1 = np.array([
            [1/2, 1/2,  0,    0],
            [  0, 1/4, 1/2, 1/4],
            [1/2, 1/3, 1/6,   0]
        ])

        self.secrets2 = Secrets(['x1','x2','x3','x4'], [1/3,1/3,0,1/3])
        self.channel2 = [
            [1/2, 1/6, 1/3,   0],
            [  0, 1/3, 2/3,   0],
            [  0, 1/2,   0, 1/2],
            [1/4, 1/4, 1/2,   0]
        ]

    def test_valid_channels(self):
        np.testing.assert_array_equal(self.channel1, np.array([
            [1/2, 1/2,  0,    0],
            [  0, 1/4, 1/2, 1/4],
            [1/2, 1/3, 1/6,   0]
        ]))

        np.testing.assert_array_equal(self.channel2, np.array([
            [1/2, 1/6, 1/3,   0],
            [  0, 1/3, 2/3,   0],
            [  0, 1/2,   0, 1/2],
            [1/4, 1/4, 1/2,   0]
        ]))

        ch1 = Channel(self.secrets1, ['y1','y2','y3','y4'], self.channel1)
        ch2 = Channel(self.secrets2, ['y1','y2','y3','y4'], self.channel2)
        self.assertListEqual(ch1.outputs, ['y1','y2','y3','y4'])
        self.assertListEqual(ch2.outputs, ['y1','y2','y3','y4'])
        self.assertEqual(ch1.num_outputs, 4)
        self.assertEqual(ch2.num_outputs, 4)

    def test_invalid_channels(self):
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3'], self.channel1)
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3','y4'], [[],[],[],[]])
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3','y4'], np.array([
                [1/2, 1/6, 1/3],
                [  0, 1/3, 2/3],
                [  0, 1/2,   0],
                [1/4, 1/4, 1/2]
            ]))
        with self.assertRaises(Exception):
            Channel(Secrets(['x1','x2'],[1/2,1/2]), ['y1','y2','y3'], self.channel1)
        with self.assertRaises(Exception):
            Channel(Secrets(['x1','x2','x3','x4'],[1,0,0,0]), ['y1','y2','y3'], self.channel1)
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3'], np.array([
                [1/2, 1/2,  0,    1/2],
                [  0, 1/4, 1/2, 1/4],
                [1/2, 1/3, 1/6,   0]
            ]))
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3'], np.array([
                [1/2, 1/2,  0,    0],
                [  0, 1/4, 1/2, 1/4],
                [  0,   0,   0,   0]
            ]))
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3'], np.array([
                [1/2, 1/2,  0,     0],
                [  0, 1/4, 1/2,  1/4],
                [  1, 1/2,   0, -1/2]
            ]))

    def test_invalid_types(self):
        with self.assertRaises(Exception):
            Channel(1,1,1)
        with self.assertRaises(Exception):
            Channel(self.secrets1, ['y1','y2','y3','y4'], 'string')
        with self.assertRaises(Exception):
            Channel(self.secrets1, 42, self.channel1)
        with self.assertRaises(Exception):
            Channel(lambda x : 'x' + str(x), ['y1','y2','y3','y4'], self.channel1)

if __name__ == '__main__':
    unittest.main()