import numpy as np
import cvxpy as cp

import logging

from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.problems.problem import Problem
from cvxpy.expressions.variable import Variable

from numpy.random import default_rng

class BipartiteIntegralOMD():

    def __init__(self, sources, cache_sizes, N, T, weights=None, cache_orders=None, step_size=None, seed=None):
        
        self.rng = default_rng(seed)
        self.e = self.rng.random()

        if weights is None:
            weights = np.ones((len(sources), len(cache_sizes), N))

        self.weights = weights

        self.max_cost = np.amax(self.weights)

        self.sources = sources
        self.cache_sizes = cache_sizes
        self.N = N
        self.T = T
        self.t = 0
        if step_size is None:
            step_size = np.sqrt((2 * np.log(N / np.sum(cache_sizes))) / (np.max(weights) * 1 * self.T))
        self.step_size = step_size

        self.yts = np.ones((len(cache_sizes), N)) / (len(cache_sizes) + N)
        self.cache_state = [set()] * len(self.cache_sizes)

        if cache_orders is None:
            cache_orders = list()
            for _ in range(len(sources)):
                cache_orders.append(self.rng.permutation(range(len(cache_sizes))))

        self.cache_orders = cache_orders


        # Precompute projection problem to improve performance
        # proj [ argmin ( sum_i x_i * log(x_i/y_i) ) ]
        self.xt_var = Variable(N, nonneg=True)
        self.yt_param = Parameter(N, nonneg=True)
        constraints = [
            cp.sum(self.xt_var) == self.cache_sizes[0],
            self.xt_var <= 1
        ]
        self.prob = Problem(Minimize(cp.sum(cp.rel_entr(self.xt_var, self.yt_param))), constraints) 

        self.hit_ratio_hist = []
        self.hits = 0
        self.req_hist = []
        self.req_count = 0

        self.total_util = 0
        self.util_hist = []

        self.name = "Bipartite Integral OMD"

    def request(self, reqs): 
        """
            Process a request (dict), which consists of one set of requests per source
        """
        self.t += 1

        req_count = 0

        for i in reqs:
            
            for _ in reqs[i]:

                req_count += 1

        self.total_util += self.get_cost(reqs) / (req_count)
        self.util_hist.append(self.total_util / self.t)

        yts_new = self.yts * np.exp(self.step_size * self.get_util_grad(reqs))

        self.yts = self.proj(yts_new)

        self.cache_state = self.online_couple_round(self.yts)

    def proj(self, yts):
        """
            Poject each cache in the network
        """
    
        yts_new = np.zeros((len(self.cache_sizes), self.N))

        for j in range(len(self.cache_sizes)):
            
            yts_new[j, :] = self.proj_single(yts[j, :])

        return yts_new 

    def proj_single(self, yt_next):
        """
            Project a single cache to a valid state
        """

        self.yt_param.value = yt_next
        try:
            self.prob.solve()
        except cp.error.SolverError:
            logging.warn("ECOS Solver failed, trying SCS...")
            self.prob.solve(solver=cp.SCS)
        return self.xt_var.value

    def online_couple_round(self, yts):
        """
            Perform online coupled rounding on each cache in the network
        """
        
        yts_new = []

        for j in range(len(self.cache_sizes)):
            
            yts_new.append(self.online_couple_round_single(yts[j, :], self.cache_sizes[j]))

        return yts_new 

    def online_couple_round_single(self, yt, size):
        """
            Use a randomly sampeled variable `e` to perform coupled online rounding, converting a continuous cache state `xt` to a discrete one without sacrificing adversarial performance.
        """
        cache_state = set()
        yt_partial_sum = 0
        for i in range(len(yt)):
            yt_partial_sum += yt[i]
            if yt_partial_sum >= self.e + len(cache_state):
                cache_state.add(i)

        # Sanity check
        if len(cache_state) > size:
            logging.critical("Rounded cache exceeds maximum size!")
            logging.critical(str(cache_state))

        return cache_state

    def get_util(self, reqs):
        """
            Get the utility of the cache state under a given request
        """

        reqs_enc = self.get_one_hot_reqs(reqs)

        out = 0

        for n in range(self.N):

            for i in range(len(self.sources)):

                for k, j in enumerate(self.cache_orders[i]):

                    out += self.weights[i, j, n] * min(1, 
                                                       reqs_enc[i, n] * 
                                                           np.sum([ int(n in self.cache_state[j_i]) 
                                                                      for j_i in self.cache_orders[i][:k+1] ]) )

        return out

    def get_cost(self, reqs):
        """
            Get the cost of the cache state under a given request
        """
        
        reqs_enc = self.get_one_hot_reqs(reqs)
        cost = 0

        for i in range(len(self.sources)):
            for el in np.argwhere(reqs_enc[i, :]):
                n = el[0]
                cost += reqs_enc[i, n] * ( self.max_cost - np.amax( [self.weights[i, j, n] * int(n in self.cache_state[j]) for j in self.cache_orders[i]] ) )

        return cost

    def get_util_grad(self, reqs):
        """
            Get the subgradient based upon the weights, request and whether 100% of the file is already cached
        """

        reqs_enc = self.get_one_hot_reqs(reqs)
        
        out = np.zeros((len(self.cache_sizes), self.N))

        for i in range(len(self.sources)):

            for k, j in enumerate(self.cache_orders[i]):

                for el in np.argwhere(reqs_enc[i, :]):
                    n = el[0]

                    #out[j, n] += self.weights[i, j, n] * reqs_enc[i, n] * int(np.sum(self.yts[self.cache_orders[i][:k], n]) <= 1) 
                    out[j, n] += self.weights[i, j, n] * reqs_enc[i, n] * int(np.sum(
                                                                                [self.yts[j_i, n] * int(self.weights[i, j, n] > 0) for j_i in range(len(self.cache_sizes))]
                                                                              ) <= 1) 
        
        return out

    def generate_optimal_state(self):
        """
            Use the utility function to generate an optimal, static cache state for calculating regret
        """
        
        yt_opt = Variable((len(self.cache_sizes), self.N), nonneg=True)
        xt = Parameter((len(self.sources), self.N), nonneg=True)
        
        reqs_enc = np.zeros((len(self.sources), self.N))

        for reqs in self.req_hist:
            reqs_enc += self.get_one_hot_reqs(reqs)

        xt.value = reqs_enc

        # Get all the links between sources and caches
        links = list()
        for i, w_i in enumerate(self.weights):
            links.append(list())
            for j, w_i_j in enumerate(w_i):
                if np.sum(w_i_j) > 0:
                    links[i].append(j)
        
        sub_problems = list()
        constraints = list()
        for i in range(len(self.sources)):
            sub_problems.append(
                cp.scalar_product(
                    xt[i], 
                    cp.minimum(
                        cp.sum(yt_opt[links[i]], axis=0), 
                        np.ones(self.N)
                    )
                )
            )

        for j, k in enumerate(self.cache_sizes):
            constraints.append(cp.sum(yt_opt[j]) <= k)
            
        prob = Problem(Maximize(cp.sum(cp.hstack(sub_problems))), constraints)
        
        try:
            prob.solve()
        except cp.error.SolverError:
            logging.warn("ECOS Solver failed, trying SCS...")
            prob.solve(solver=cp.SCS)

        return self.online_couple_round(yt_opt.value)


    def get_one_hot_reqs(self, reqs):
        """
            Convert the dictionary/set representation of a request batch to a one-hot encoded version for easier computation
        """

        reqs_one_hot = np.zeros((len(self.sources), self.N))

        for src in reqs:
            reqs_one_hot[src, :] = np.array([int(i in reqs[src]) for i in range(self.N)])

        return reqs_one_hot
