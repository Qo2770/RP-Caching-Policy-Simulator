import numpy as np

from cache.omd_cache import OMD

class BipartiteOMDNaive():

    def __init__(self, sources, cache_sizes, N, T, weights=None):

        self.sources = sources
        self.ks = cache_sizes
        self.N = N
        self.T = T
        self.t = 0

        if weights is None:
            weights = np.ones((len(sources), len(cache_sizes), N))

        self.weights = weights

        self.max_cost = np.amax(weights)

        self.caches = [OMD(self.ks[i], N, T) for i in range(len(self.ks))]

        self.hit_ratio_hist = []
        self.hits = 0
        self.req_hist = []
        self.req_count = 0

        self.total_util = 0
        self.util_hist = []

        self.name = "Naive Bipartite OMD"

    def request(self, reqs):

        self.t += 1

        hits = 0
        req_count = 0

        for src_req in reqs:

            for n in reqs[src_req]:

                submitted = False

                for j, w in enumerate(np.argpartition(self.weights[src_req, : , n], len(self.sources)-1)):

                    if w == 0:
                        continue
                    
                    if n in self.caches[j].state:
                        hits += int(self.caches[j].request(n))
                        submitted = True
                        break

                if not submitted:
                    for j, w in enumerate(np.argpartition(self.weights[src_req, : , n], len(self.sources)-1)):

                        if w == 0:
                            continue

                        self.caches[j].request(n)
                        break

                req_count += 1

        self.req_hist.append(reqs)
        self.req_count += req_count
        hit_ratio = self.hits / self.req_count
        self.hit_ratio_hist.append(hit_ratio)
        self.hits += hits

        self.total_util += (req_count * self.max_cost - hits) / len(reqs)
        self.util_hist.append(self.total_util / self.t)

    def generate_optimal_state(self):
        raise NotImplemented("Cannot generate optimal cache state")

        
