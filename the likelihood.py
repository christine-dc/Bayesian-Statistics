"""
Created on Sun March 09, 2025  20:41:59

@author: Christine D. Dela Cruz
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

mu =np.linspace(1.65, 1.8,num=50)
test = np.linspace(0,2)
uniform_dist = sts.uniform.pdf(mu) + 1

def likelihood_func(datum,mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation at 1.7m")
plt.xlabel("Value of $\mu$")
plt.ylabel("Probability Density/Likelihood")

plt.show()