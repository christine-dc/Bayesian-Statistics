"""
Created on Sun March 09, 2025  20:47:59

@author: Christine D. Dela Cruz
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

mu =np.linspace(1.65, 1.8,num=50)
test = np.linspace(0,2)
uniform_dist = sts.uniform.pdf(mu) + 1

uniform_dist = uniform_dist/uniform_dist.sum()

def likelihood_func(datum,mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

import scipy as sp

unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")

plt.show()