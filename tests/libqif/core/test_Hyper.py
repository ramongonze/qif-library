from libqif.core.secrets import Secrets
from libqif.core.channel import Channel
from libqif.core.hyper import Hyper
import numpy as np
import unittest
import os

class TestHyper(unittest.TestCase):
    def setUp(self):
        self.prior1 = np.array([1/4, 1/2, 1/4])
        self.channel1 = np.array([
            [1/2, 1/2,  0,    0],
            [  0, 1/4, 1/2, 1/4],
            [1/2, 1/3, 1/6,   0]
        ])

        self.prior2 = np.array([1/3,1/3,0,1/3])
        self.channel2 = np.array([
            [1/2, 1/6, 1/3,   0],
            [  0, 1/3, 2/3,   0],
            [  0, 1/2,   0, 1/2],
            [1/4, 1/4, 1/2,   0]
        ])

        self.prior3 = np.array([1/4,1/4,1/4,1/4])
        self.channel3 = np.array([
            [1/2, 1/2,   0,   0],
            [  0,   0,   1,   0],
            [1/2, 1/4,   0, 1/4],
            [1/8, 1/8, 1/4, 1/2]
        ])

        self.channel_identity_3 = np.identity(3)
        self.channel_identity_4 = np.identity(4)

    def test_valid_hypers(self):
        secrets = Secrets(['x1','x2','x3'], self.prior1)
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel1)
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.joint, np.array([
            [1/8,  1/8,    0,   0],
            [  0,  1/8,  1/4, 1/8],
            [1/8, 1/12, 1/24,   0],
        ]))
        np.testing.assert_array_equal(hyper.outer, np.array([1/4,1/3,7/24,1/8]))
        np.testing.assert_array_equal(hyper.inners, np.array([
            [1/2, 3/8,   0,  0],
            [  0, 3/8, 6/7,  1],
            [1/2, 1/4, 1/7,  0]
        ]))

        # Channel that leaks everything
        channel = Channel(secrets, ['y1','y2','y3'], self.channel_identity_3)
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.outer, secrets.prior)
        np.testing.assert_array_equal(hyper.inners, np.identity(3))

        # Channel that leaks nothing
        channel = Channel(secrets, ['y1'], np.ones((3,1)))
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.outer, np.array([1]))
        np.testing.assert_array_equal(hyper.inners, np.array([secrets.prior]).T)

        secrets = Secrets(['x1','x2','x3','x4'], self.prior2)
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel2)
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.joint, np.array([
            [ 1/6,  1/18,  1/9,   0],
            [   0,   1/9,  2/9,   0],
            [   0,     0,    0,   0],
            [1/12,  1/12,  1/6,   0],
        ]))
        np.testing.assert_array_equal(hyper.outer, np.array([1/4,3/4]))
        np.testing.assert_array_equal(hyper.inners, np.array([
            [2/3, 2/9],
            [  0, 4/9],
            [  0,   0],
            [1/3, 1/3]
        ]))

        # Exercise 4.1 of The Science of Quantitative Information Flow book
        secrets = Secrets(['x1','x2','x3','x4'], self.prior3)
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel3)
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.outer, np.array([9/32,7/32,10/32,6/32]))
        np.testing.assert_array_equal(hyper.inners, np.array([
            [4/9, 4/7,   0,   0],
            [  0,   0, 4/5,   0],
            [4/9, 2/7,   0, 1/3],
            [1/9, 1/7, 1/5, 2/3]
        ]))

        # Channel that leaks everything
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel_identity_4)
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.outer, secrets.prior)
        np.testing.assert_array_equal(hyper.inners, np.identity(4))

        # Channel that leaks nothing
        channel = Channel(secrets, ['y1'], np.ones((4,1)))
        hyper = Hyper(channel)
        np.testing.assert_array_equal(hyper.outer, np.array([1]))
        np.testing.assert_array_equal(hyper.inners, np.array([secrets.prior]).T)

        # Exercise 4.2 of The Science of Quantitative Information Flow book
        secrets = Secrets(['x1','x2','x3','x4','x5','x6','x7','x8'], [1/8]*8)
        channel_c = Channel(secrets, ['y1','y2'], np.array([
            [1,0],
            [1,0],
            [1,0],
            [1,0],
            [1,0],
            [1,0],
            [0,1],
            [1,0],
        ]))

        channel_d = Channel(secrets, ['y1','y2','y3','y4'], np.array([
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
            [0,1,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0],
        ]))

        hyper_c = Hyper(channel_c)
        hyper_d = Hyper(channel_d)

        np.testing.assert_array_equal(hyper_c.outer, np.array([7/8,1/8]))
        np.testing.assert_array_equal(hyper_c.inners, np.array([
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [  0, 1],
            [1/7, 0],
        ]))

        np.testing.assert_array_equal(hyper_d.outer, np.array([1/2,1/4,1/8,1/8]))
        np.testing.assert_array_equal(hyper_d.inners, np.array([
            [1/4,   0,   0,   0],
            [1/4,   0,   0,   0],
            [1/4,   0,   0,   0],
            [1/4,   0,   0,   0],
            [  0, 1/2,   0,   0],
            [  0, 1/2,   0,   0],
            [  0,   0,   0,   1],
            [  0,   0,   1,   0],
        ]))

    def test_valid_prior_updates(self):
        secrets = Secrets(['x1','x2','x3'], [1,0,0])
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel1)
        hyper = Hyper(channel)
        hyper.update_prior(self.prior1)
        np.testing.assert_array_equal(hyper.joint, np.array([
            [1/8,  1/8,    0,   0],
            [  0,  1/8,  1/4, 1/8],
            [1/8, 1/12, 1/24,   0],
        ]))
        np.testing.assert_array_equal(hyper.outer, np.array([1/4,1/3,7/24,1/8]))
        np.testing.assert_array_equal(hyper.inners, np.array([
            [1/2, 3/8,   0,  0],
            [  0, 3/8, 6/7,  1],
            [1/2, 1/4, 1/7,  0]
        ]))

        # Channel that leaks everything
        secrets = Secrets(['x1','x2','x3'], [1,0,0])
        channel = Channel(secrets, ['y1','y2','y3'], self.channel_identity_3)
        hyper = Hyper(channel)
        hyper.update_prior(self.prior1)
        np.testing.assert_array_equal(hyper.outer, secrets.prior)
        np.testing.assert_array_equal(hyper.inners, np.identity(3))

        # Channel that leaks nothing
        secrets = Secrets(['x1','x2','x3'], [1,0,0])
        channel = Channel(secrets, ['y1'], np.ones((3,1)))
        hyper = Hyper(channel)
        hyper.update_prior(self.prior1)
        np.testing.assert_array_equal(hyper.outer, np.array([1]))
        np.testing.assert_array_equal(hyper.inners, np.array([secrets.prior]).T)

        secrets = Secrets(['x1','x2','x3','x4'], [1,0,0,0])
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel2)
        hyper = Hyper(channel)
        hyper.update_prior(self.prior2)
        np.testing.assert_array_equal(hyper.joint, np.array([
            [ 1/6,  1/18,  1/9,   0],
            [   0,   1/9,  2/9,   0],
            [   0,     0,    0,   0],
            [1/12,  1/12,  1/6,   0],
        ]))
        np.testing.assert_array_equal(hyper.outer, np.array([1/4,3/4]))
        np.testing.assert_array_equal(hyper.inners, np.array([
            [2/3, 2/9],
            [  0, 4/9],
            [  0,   0],
            [1/3, 1/3]
        ]))

        # Exercise 4.1 of The Science of Quantitative Information Flow book
        secrets = Secrets(['x1','x2','x3','x4'], [1,0,0,0])
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel3)
        hyper = Hyper(channel)
        hyper.update_prior(self.prior3)
        np.testing.assert_array_equal(hyper.outer, np.array([9/32,7/32,10/32,6/32]))
        np.testing.assert_array_equal(hyper.inners, np.array([
            [4/9, 4/7,   0,   0],
            [  0,   0, 4/5,   0],
            [4/9, 2/7,   0, 1/3],
            [1/9, 1/7, 1/5, 2/3]
        ]))

        # Channel that leaks everything
        secrets = Secrets(['x1','x2','x3','x4'], [1,0,0,0])
        channel = Channel(secrets, ['y1','y2','y3','y4'], self.channel_identity_4)
        hyper = Hyper(channel)
        hyper.update_prior(self.prior3)
        np.testing.assert_array_equal(hyper.outer, secrets.prior)
        np.testing.assert_array_equal(hyper.inners, np.identity(4))

        # Channel that leaks nothing
        secrets = Secrets(['x1','x2','x3','x4'], [1,0,0,0])
        channel = Channel(secrets, ['y1'], np.ones((4,1)))
        hyper = Hyper(channel)
        hyper.update_prior(self.prior3)
        np.testing.assert_array_equal(hyper.outer, np.array([1]))
        np.testing.assert_array_equal(hyper.inners, np.array([secrets.prior]).T)

        # Exercise 4.2 of The Science of Quantitative Information Flow book
        secrets = Secrets(['x1','x2','x3','x4','x5','x6','x7','x8'], [1,0,0,0,0,0,0,0])
        channel_c = Channel(secrets, ['y1','y2'], np.array([
            [1,0],
            [1,0],
            [1,0],
            [1,0],
            [1,0],
            [1,0],
            [0,1],
            [1,0],
        ]))

        channel_d = Channel(secrets, ['y1','y2','y3','y4'], np.array([
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
            [1,0,0,0],
            [0,1,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0],
        ]))

        hyper_c = Hyper(channel_c)
        hyper_c.update_prior([1/8]*8)
        hyper_d = Hyper(channel_d)
        hyper_d.update_prior([1/8]*8)

        np.testing.assert_array_equal(hyper_c.outer, np.array([7/8,1/8]))
        np.testing.assert_array_equal(hyper_c.inners, np.array([
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [1/7, 0],
            [  0, 1],
            [1/7, 0],
        ]))

        np.testing.assert_array_equal(hyper_d.outer, np.array([1/2,1/4,1/8,1/8]))
        np.testing.assert_array_equal(hyper_d.inners, np.array([
            [1/4,   0,   0,   0],
            [1/4,   0,   0,   0],
            [1/4,   0,   0,   0],
            [1/4,   0,   0,   0],
            [  0, 1/2,   0,   0],
            [  0, 1/2,   0,   0],
            [  0,   0,   0,   1],
            [  0,   0,   1,   0],
        ]))

if __name__ == '__main__':
    unittest.main()
    