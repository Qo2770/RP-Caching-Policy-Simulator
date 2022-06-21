from .rounding_regret_cache import RoundingCache

import cvxpy as cp
import numpy as np
from numpy.random import default_rng

import logging

class OMD(RoundingCache):
    
    def __init__(self, size, N, T, weights=None, step_size=None, seed=None):
        super().__init__(size, N, T, seed)
    
        # If weights are None, assume 1 for all
        if weights is None:
            weights = np.ones(N)
        self.weights = weights
        
        # Initial cache parameters that ensure valid start state
        self.xt = np.ones(N) / N
        self.P = 1
        if step_size is None:
            step_size = np.sqrt((2 * np.log(N / size)) / (np.max(weights) * 10))
        self.step_size = step_size

        # Precompute projection problem to improve performance
        # proj [ argmin ( sum_i x_i * log(x_i/y_i) ) ]
        self.xt_var = cp.Variable(N, nonneg=True)
        self.yt_param = cp.Parameter(N, nonneg=True)
        constraints = [
            cp.sum(self.xt_var) == self.size,
            self.xt_var <= 1
        ]
        self.prob = cp.Problem(cp.Minimize(cp.sum(cp.rel_entr(self.xt_var, self.yt_param))), constraints)

        self.name = "OMD"

    def request(self, req):
        """
            Process a request. Uses mirror descent to determine optimal cache state.
            Returns True for a cache hit, False otherwise.
        """
        super().request(req) 
        
        # Stores whether the request is in the cache intially
        found = False
        if req in self.state:
            found = True
            self.hits += 1
        
        # Multiplicative change towards the gradient
        self.yt_next = self.xt * np.exp(self.step_size * np.array(self.get_util_grad(req))) 
        
        # Project to a valid state, and create discrete state from that
        self.xt = self.proj_slow(self.yt_next)
        self.state = self.online_couple_round(self.xt)

        # Update regret
        self.regret = self.calc_regret()
        self.regret_hist.append(self.regret)

        return found

    def get_util_grad(self, req):
        """
            Get the gradient of the utility function for a given request.
        """
        out = np.zeros(self.N)
        out[req] = self.weights[req]
        return out

    def proj_slow(self, yt_next):
        """
            Project to a valid cache state. This variant uses CVXPY, which is substantially slower than alternative algorithms.
        """
        self.yt_param.value = yt_next
        try:
            self.prob.solve()
        except cp.error.SolverError:
            logging.warn("ECOS Solver failed, trying SCS...")
            self.prob.solve(solver=cp.SCS)
        return self.xt_var.value

