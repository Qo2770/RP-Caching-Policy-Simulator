from .rounding_regret_cache import RoundingCache

import logging

import cvxpy as cp
import numpy as np

from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem
from numpy.random import default_rng

class BipartiteOMD():

    def __init__(self, sizes, sources, weights, N, T, seed):

        self.e = default_rng(seed).random()

        self.ks = np.array(sizes)
        self.N = N
        self.T = T
        self.t = 1
        self.sources = sources
        self.weights = weights

        self.z = np.ones((len(sources), len(sizes), self.N)) / (len(sources) + len(sizes) + N)
        self.xts = np.ones((len(sizes), self.N)) / (len(sizes) + N)
        
        # FIXME: d is the max degree of any cache, so this should be calculated
        self.d = 10

        #self.const_step_size = (len(sources) ** (3/4)) / ((2 * self.d * (np.log((N / sizes[0])) + 1)) ** (1/4))
        
        self.step_size = np.sqrt((2 * np.log(N / sizes[0])) / (np.max(weights) * 10))

        self.hits = 0
        self.req_hist = []
        self.hit_ratio_hist = []
        self.req_count = 0

        self.name = "Bipartite Fractional OMD"
    
    def request(self, reqs):
        """
            reqs: A dict, with the key being the index of the cache, and the value being the set of requested items

        """

        hits = 0
        req_count = 0

        for i in reqs:

            for n in reqs[i]:

                n_frac = 0
                req_count += 1

                for j in range(len(self.ks)):

                    if self.weights[i, j, n] * self.ks[j] > 0:

                        n_frac += self.weights[i, j, n] * self.ks[j]
                        if n_frac > 1:
                            break
                        hits += self.weights[i, j, n] * self.ks[j]

        self.req_hist.append(reqs)
        self.req_count += req_count
        hit_ratio = self.hits / self.req_count
        self.hit_ratio_hist.append(hit_ratio)
        self.hits += hits

        self.z = self.z * np.exp(self.step_size * self.get_util_grad(reqs))

        self.z, self.xts = self.proj(self.z, self.xts) 
        
        self.t += 1

    def get_util_grad(self, reqs):
        """
            Get the gradient of the utility
        """
        
        out = np.zeros((len(self.sources), len(self.ks), self.N))
        
        # Loop through all sources in reqs
        for i_r in reqs:

            n_r = np.array([i in reqs[i_r] for i in range(self.N)])
            
            for j in range(len(self.ks)):

                out += n_r * self.weights[i_r, j]

        return out

    def get_util(self, reqs):
        """
            Get the utility
        """

        util = 0
        
        for j in range(len(self.ks)):
            
            util += np.sum(self.z[:, j, :] * self.weights[:, j, :] * reqs)

        return util

    def get_one_hot_reqs(self, reqs):
        """
            Create a one-hot-encoded version of the request to make calculations easier
        """

        reqs_one_hot = np.zeros((len(self.sources), self.N))

        for src in reqs:
            reqs_one_hot[src] = np.array([int(i in reqs[src]) for i in range(self.N)])

        return reqs_one_hot

    def proj(self, z, xt):
        """
            Project caches onto a valid state
        """

        xt_var = Variable((len(self.ks), self.N), nonneg=True)
        
        xt_var.value = xt


        constraints = [
            cp.sum(xt_var, axis=1) == self.ks,
            xt_var <= 1,
        ]

        zt_var = []
        prob_var = []
        
        for src in range(len(self.sources)):
            zt_var.append(Variable((len(self.ks), self.N), nonneg=True))
            constraints.append(zt_var[src] <= 1)
            constraints.append(zt_var[src] <= xt_var)
            prob_var.append(cp.rel_entr(zt_var[src], z[src]))
        
        prob = Problem(Minimize(cp.sum(cp.hstack(prob_var))), constraints)
 
        try:
            prob.solve()
        except cp.error.SolverError:
            logging.warn("ECOS Solver failed, trying SCS...")
            prob.solve(solver=cp.SCS)

        zt_out = [i.value for i in zt_var]

        return np.array(zt_out), xt_var.value

    def generate_optimal_state(self):
        raise NotImplemented("Cannot generate optimal cache state")


