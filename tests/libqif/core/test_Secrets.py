from libqif.core.secrets import Secrets
import numpy as np
import unittest
import os

class TestSecrets(unittest.TestCase):
    def setUp(self):
        # X = {x1}
        self.priors1 = [
            [1]
        ]

        self.invalid_priors1 = [
            [0],
            [2],
            [0.5],
            [-1]
        ]

        # X = {x1,x2}
        self.priors2 = [
            [1,0],
            [0,1],
            [1/2,1/2]
        ]

        self.invalid_priors2 = [
            [0,0],
            [1,1],
            [0.5,0.3],
            [1.5,-0.5],
            [-0.5,-0.5]
        ]

        # X = {x1,x2,x3}
        self.priors3 = [
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1/3,1/3,1/3],
            [1/2,1/2,0],
            [1/2,0,1/2],
            [0,1/2,1/2],
        ]

        self.invalid_priors3 = [
            [0,0,0],
            [1,1,1],
            [0.5,0.3,0],
            [1.5,0,-0.5],
            [-0.5,-0.5,-1],
            [0.5,0.5,-1]
        ]

        self.labels1 = ['x1']
        self.labels2 = ['x1','x2']
        self.labels3 = ['x1','x2','x3']

    def test_valid_priors_and_labels(self):
        for i in np.arange(len(self.priors1)):
            secrets = Secrets(self.labels1, self.priors1[i])
            self.assertListEqual(secrets.labels, ['x1'])
            self.assertEqual(secrets.num_secrets, 1)
            np.testing.assert_array_equal(secrets.prior, self.priors1[i])

        for i in np.arange(len(self.priors2)):
            secrets = Secrets(self.labels2, self.priors2[i])
            self.assertListEqual(secrets.labels, ['x1','x2'])
            self.assertEqual(secrets.num_secrets, 2)
            np.testing.assert_array_equal(secrets.prior, self.priors2[i])

        for i in np.arange(len(self.priors3)):
            secrets = Secrets(self.labels3, self.priors3[i])
            self.assertListEqual(secrets.labels, ['x1','x2','x3'])
            self.assertEqual(secrets.num_secrets, 3)
            np.testing.assert_array_equal(secrets.prior, self.priors3[i])

    def test_invalid_labels_size(self):
        for i in np.arange(len(self.priors1)):
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels2, self.priors1[i])
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels3, self.priors1[i])
        
        for i in np.arange(len(self.priors2)):
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels1, self.priors2[i])
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels3, self.priors2[i])

        for i in np.arange(len(self.priors3)):
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels1, self.priors3[i])
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels2, self.priors3[i])
            
    def test_invalid_prior_distribution(self):
        for prior in self.invalid_priors1:
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels1, prior)
        
        for prior in self.invalid_priors2:
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels2, prior)
        
        for prior in self.invalid_priors3:
            with self.assertRaises(Exception):
                secrets = Secrets(self.labels3, prior)

if __name__ == '__main__':
    unittest.main()