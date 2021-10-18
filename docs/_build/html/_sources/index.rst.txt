.. QIF Library documentation master file, created by
   sphinx-quickstart on Mon Oct 18 18:58:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QIF Library's documentation
===========================

This library models the basics of Quantitative Information flow (QIF) such as
secrets, channels, and the :math:`g`-vulnerability framework. The library is
object oriented and the main functionalities are described in the documentation
of each class.

Here are some examples of usage: ::
   
   from libqif.core.secrets import Secrets
   from libqif.core.channel import Channel
   from libqif.core.hyper import Hyper
   from libqif.core.gvulnerability import Gain
   import numpy as np

   secrets = Secrets(['x1','x2','x3','x4'], [1/3, 1/3, 0, 1/3])
   channel = Channel(secrets, ['y1','y2','y3','y4'], np.array([
      [1/2, 1/6, 1/3,   0],
      [  0, 1/3, 2/3,   0],
      [  0, 1/2,   0, 1/2],
      [1/4, 1/4, 1/2,   0]
   ]))
   hyper = Hyper(channel)
   gain = Gain(secrets, ['w1','w2','w3','w4'], np.identity(4)) # Bayes vulnerability

   print('Prior distribution: ' + str(secrets.prior))
   print('Channel:\n' + str(channel.matrix))
   print('\nOuter distribution: ' + str(hyper.outer))
   print('Inner distributions:\n' + str(hyper.inners))
   print('\nPrior Bayes vulnerability: ' + str(gain.prior_vulnerability()))
   print('Posterior Bayes vulnerability: ' + str(gain.posterior_vulnerability(hyper)))

Output: ::

   Prior distribution: [0.33333333 0.33333333 0.         0.33333333]
   Channel:
   [[0.5        0.16666667 0.33333333 0.        ]
   [0.         0.33333333 0.66666667 0.        ]
   [0.         0.5        0.         0.5       ]
   [0.25       0.25       0.5        0.        ]]
   
   Outer distribution: [0.25 0.75]   
   Inner distributions:
   [[0.66666667 0.22222222]
   [0.         0.44444444]
   [0.         0.        ]
   [0.33333333 0.33333333]]

   Prior Bayes vulnerability: 0.3333333333333333
   Posterior Bayes vulnerability: 0.5

.. toctree::
   :maxdepth: 3
   :caption: Packages:

   modules/libqif/core
   modules/libqif/util

.. highlight:: python

References
----------
*Alvim, Mário S., Konstantinos Chatzikokolakis, Annabelle McIver,
Carroll Morgan, Catuscia Palamidessi, and Geoffrey Smith.
The Science of Quantitative Information Flow. Springer International
Publishing, 2020.*