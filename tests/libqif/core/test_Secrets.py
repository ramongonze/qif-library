from libqif.core.secrets import Secrets
import numpy as np
import unittest
import os

class TestSecrets(unittest.TestCase):
    def setUp(self):
        # X = {x1,x2}
        self.priors1 = [
            [1,0],
            [0,1],
            [1/2,1/2]
        ]

        self.invalid_priors1 = [
            [0,0],
            [1,1],
            [0.5,0.3],
            [1.5,-0.5],
            [-0.5,-0.5]
        ]

        # X = {x1,x2,x3}
        self.priors2 = [
            [1,0,0],
            [0,1,0],
            [0,0,1],
            [1/3,1/3,1/3],
            [1/2,1/2,0],
            [1/2,0,1/2],
            [0,1/2,1/2],
        ]

        self.invalid_priors2 = [
            [0,0,0],
            [1,1,1],
            [0.5,0.3,0],
            [1.5,0,-0.5],
            [-0.5,-0.5,-1],
            [0.5,0.5,-1]
        ]

        self.labels1 = ['x1','x2']
        self.labels2 = ['x1','x2','x3']

    def test_valid_priors_and_labels(self):
        for prior in self.priors1:
            secrets = Secrets(self.labels1, prior)
            self.assertListEqual(secrets.labels, ['x1','x2'])
            self.assertEqual(secrets.num_secrets, 2)
            np.testing.assert_array_equal(secrets.prior, prior)

        for prior in self.priors2:
            secrets = Secrets(self.labels2, prior)
            self.assertListEqual(secrets.labels, ['x1','x2','x3'])
            self.assertEqual(secrets.num_secrets, 3)
            np.testing.assert_array_equal(secrets.prior, prior)

    def test_invalid_labels_size(self):
        with self.assertRaises(Exception):
            Secrets(['x1'], [1])

        for prior in self.priors1:
            with self.assertRaises(Exception):
                Secrets(self.labels2, prior)
        
        for prior in self.priors2:
            with self.assertRaises(Exception):
                Secrets(self.labels1, prior)
            
    def test_invalid_prior_distribution(self):
        for prior in self.invalid_priors1:
            with self.assertRaises(Exception):
                Secrets(self.labels1, prior)
        
        for prior in self.invalid_priors2:
            with self.assertRaises(Exception):
                Secrets(self.labels2, prior)

    def test_invalid_labels_type(self):
        with self.assertRaises(Exception):
            Secrets('invalid type', [1])
        with self.assertRaises(Exception):
            Secrets(42, [1])
        with self.assertRaises(Exception):
            Secrets({'x1':1,'x2':2}, [1])
        
    def test_invalid_prior_type(self):
        with self.assertRaises(Exception):
            Secrets(['x1','x2','x3'], 42)
        with self.assertRaises(Exception):
            Secrets(['x1','x2','x3'], 'string')
        with self.assertRaises(Exception):
            Secrets(['x1','x2'], {'x1':0.5, 'x2':0.5})

    def test_valid_prior_updates(self):
        secrets1 = Secrets(self.labels1, [1,0])
        secrets2 = Secrets(self.labels2, [1,0,0])

        for prior in self.priors1:
            secrets1.update_prior(prior)
            self.assertListEqual(secrets1.labels, ['x1','x2'])
            self.assertEqual(secrets1.num_secrets, 2)
            np.testing.assert_array_equal(secrets1.prior, prior)

        for prior in self.priors2:
            secrets2.update_prior(prior)
            self.assertListEqual(secrets2.labels, ['x1','x2','x3'])
            self.assertEqual(secrets2.num_secrets, 3)
            np.testing.assert_array_equal(secrets2.prior, prior)
    
    def test_invalid_prior_update(self):
        secrets1 = Secrets(self.labels1, [1,0])
        secrets2 = Secrets(self.labels2, [1,0,0])

        for prior in self.invalid_priors1:
            with self.assertRaises(Exception):
                secrets1.update_prior(prior)
        
        for prior in self.invalid_priors2:
            with self.assertRaises(Exception):
                secrets2.update_prior(prior)

        with self.assertRaises(Exception):
            secrets1.update_prior(42)
        with self.assertRaises(Exception):
            secrets1.update_prior('string')
        with self.assertRaises(Exception):
            secrets1.update_prior({'x1':0.5, 'x2':0.5})
            
        with self.assertRaises(Exception):
            secrets2.update_prior(42)
        with self.assertRaises(Exception):
            secrets2.update_prior('string')
        with self.assertRaises(Exception):
            secrets2.update_prior({'x1':0.5, 'x2':0.5})

if __name__ == '__main__':
    unittest.main()