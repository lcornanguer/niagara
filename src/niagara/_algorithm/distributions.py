"""
Distributions classes (implemented: exponential and lognormal)
"""

import math

import numpy as np
# from matplotlib import pyplot as plt
from scipy.stats import expon, lognorm, stats, weibull_min, gamma

from .options import args
from .encoding import universal_integer_encoding, universal_real_encoding

class Distribution:
    def __init__(self, alpha=None, rng=None):
        self.alpha = alpha
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    def get_parameter_cost(self):
        if self.alpha is None: return 0
        return universal_real_encoding(self.alpha, args.precision)


class ExponentialDistribution(Distribution):
    def __init__(self, intensity=None, alpha=None, rng=None):
        super().__init__(rng=rng)
        self.intensity = intensity
        self.alpha = alpha

    def fit_to_delays(self, delays):
        """
        Computes lambda and alpha (if np.inf in delays) parameters for a list of delays.
        :param delays: list of delays (can be np.inf)
        :return: estimated lambda
        """
        triggered = [d for d in delays if d != np.inf]
        if len(triggered) == 0:
            self.intensity = 0
        else:
            intensity_est = 1 / np.mean(triggered) # 1/expon.fit([d for d in delays if d != np.inf], floc=0)[1]
            self.intensity = min(intensity_est, 10**99) if not np.isnan(intensity_est) else 0
        if len(delays) == 0: return 0
        self.alpha = len(triggered) / len(delays)
        return self.intensity

    def generate_delay(self):
        scale = 1 / self.intensity
        return self.rng.exponential(scale)

    def cost_per_delay(self, delays):
        """
        Return the cost of delays that are not infinite
        :param delays: list of delays (can contain np.inf but won't be used)
        :return: cost per delay
        """
        if self.intensity == 0: return [np.inf for _ in delays]

        fired = np.array([d for d in delays if d != np.inf])
        probs = expon.pdf(fired, scale=1 / self.intensity)

        probs = np.clip(probs, 10 ** -99, 1)  # avoid log(0)
        delay_costs = -np.log2(probs)

        if self.alpha is not None and len(fired) > 0:
            delay_costs -= np.log2(self.alpha)

        return delay_costs

    def __repr__(self) -> str:
        return f"ExponentialDistrib(lambda: {self.intensity}, alpha: {self.alpha})"

    def skipped_cost(self, delays):
        """
        Return the cost of the infinite delays (for parent events that are not matched)
        :param delays: all delays
        :return: cost
        """
        skipped = [d for d in delays if d == np.inf]
        skips = len(skipped)
        return skips * -np.log2(1-self.alpha) if skips > 0 and self.alpha > 0 else 0

    def get_parameter_cost(self):
        cost = super().get_parameter_cost()
        cost += universal_real_encoding(self.intensity, args.precision) if self.intensity is not None else 0
        return cost



class LogNormalDistribution(Distribution):
    def __init__(self, sigma=None, mu=None, alpha=None, rng=None):
        super().__init__(rng=rng)
        self.sigma = sigma  # shape
        self.mu = mu  # mean in log space
        self.alpha = alpha

    def fit_to_delays(self, data):
        shape, loc, scale = lognorm.fit([max(d, 10**-99) for d in data], floc=0) # floc cannot be >= than the min value
        self.mu = np.log(scale) # scale = exp(mu)
        self.sigma = shape

    def cost_per_delay(self, delays):
        fired = np.array([d for d in delays if d != np.inf])
        if len(fired) == 0:
            return []

        sigma = self.sigma
        scale = np.exp(self.mu)
        loc = 0

        probs = lognorm.pdf(fired, sigma, loc=loc, scale=scale)

        probs = np.clip(probs, 10 ** -99, 1.0)  # Prevent log(0)
        delay_cost = -np.log2(probs)

        return delay_cost

    def generate_delay(self):
        return lognorm.rvs(self.sigma, loc=0, scale=np.exp(self.mu), random_state=self.rng) # self.rng.lognormal(self.mu, self.sigma) #

    def get_parameter_cost(self):
        cost = super().get_parameter_cost()
        cost += universal_real_encoding(self.sigma, args.precision)
        cost += universal_real_encoding(self.mu, args.precision)
        return cost

    def __repr__(self) -> str:
        return f"LogNormalDistrib(sigma: {self.sigma}, mu: {self.mu})" # alpha always one
