from .rounding_regret_cache import RoundingCache

import cvxpy as cp
import numpy as np

import logging

class OGA(RoundingCache):

    def __init__(self, size, N, T, step_size=1, seed=None, weights=None):
        super().__init__(size, N, T, seed)

        self.step_size = step_size
        
        # The internal cache state, which is continuous
        self.yt = np.zeros(N)
        
        # If no weights are provided, assume 1 for all weights
        if weights is None:
            weights = np.ones(N)
        self.weights = weights

        # Setup the problem for projection
        self.yt_param = cp.Parameter(N, nonneg=True)
        self.yt_proj_var = cp.Variable(N, nonneg=True)
        constraints = [
            cp.sum(self.yt_proj_var) == self.size,
            self.yt_proj_var <= 1
        ]
        self.projection_prob = cp.Problem(cp.Minimize(cp.sum_squares(self.yt_param - self.yt_proj_var)), constraints)

        self.name = "OGA"

    def request(self, req):
        """
            Process a request. Uses gradient descent to determine optimal cache state.
            Returns True for a cache hit, False otherwise.
        """
        super().request(req)
        
        # Stores whether the request is in the cache initially
        found = False

        if req in self.state:
            # Cache hit
            self.hits += 1
            found = True
        
        # Gradient descent. Nudge yt in the direction of the gradient by step_size, then project to a valid state and regen the actual, discrete cache from yt
        self.yt[req] = self.yt[req] + self.step_size * self.get_utility_gradient(req)
        self.yt = self.project()
        self.state = self.online_couple_round(self.yt)
        
        # Update regret
        self.regret = self.calc_regret()
        self.regret_hist.append(self.regret)

        return found

    def get_utility_with_state(self, req, state):
        """
            Get the utility. This is just the basic utility with a weight applied.
        """
        return self.weights[req] * super().get_utility_with_state(req, state)

    def get_utility_gradient(self, req):
        """
            Get the gradient of the utility function.
        """
        return self.weights[req]

    def project(self):
        """
            Project a potentially invalid cache state (meaning one that exceeds the cache size) to a valid one.
        """
        self.yt_param.value = self.yt
        self.projection_prob.solve()
        if self.projection_prob.status != "optimal":
            raise Exception("CVXPY could not project onto a valid cache state. Status was", self.projection_prob.status)
        return self.yt_proj_var.value

