"""
Created on Sun March 09, 2025  20:21:15

@author: Christine D. Dela Cruz
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

mu =np.linspace(1.65, 1.8,num=50)
test = np.linspace(0,2)
uniform_dist = sts.uniform.pdf(mu) + 1

uniform_dist = uniform_dist/uniform_dist.sum()
beta_dist = sts.beta.pdf(mu, 2, 5, loc = 1.65, scale = 0.2)
beta_dist = beta_dist/beta_dist.sum()
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.legend()

plt.show()