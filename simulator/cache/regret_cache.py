from .cache import Cache

import numpy as np

class RegretCache(Cache):

    def __init__(self, size, N, T):
        super().__init__(size, N, T)

        self.regret = 0
        self.regret_hist = list()
        self.utility_hist = list()

        # How often each element is requested. Saved separetly to improve performance.
        self.req_freq = dict()

        self.name = "Regret Cache"

    def request(self, req):
        """
            Process a request. Updates the request frequency and utility history, but does not change state.
        """
        super().request(req)
        if not (req in self.req_freq):
            self.req_freq[req] = 1
        else:
            self.req_freq[req] += 1
        self.utility_hist.append(self.get_utility(req))

    def get_utility(self, req):
        """
            Get the utility of the state-request combination.
        """
        return self.get_utility_with_state(req, self.state)

    def get_utility_with_state(self, req, state):
        """
            Basic utility from a given request and state. If the state contains the requested item, return 1, otherwise 0.
        """
        if req in state:
            return 1
        return 0

    def get_utility_gradient(self, req):
        """
            Get the gradient of the utility function.
        """
        return 1

    def gen_ideal_cache(self):
        """
            Generate the ideal, static cache in hindsight. This is achieved by simply adding the most requested items until the cache is full.
        """
        # The indices of the largest values in the dict
        freqs = np.argpartition(list(self.req_freq), -min(self.size, len(self.req_freq)))[-min(self.size, len(self.req_freq)):]
        req_keys = np.array(list(self.req_freq.keys()))
        return set(req_keys[freqs])

    def calc_regret(self):
        """
            Calculate the regret metric, which is the utility of state of the cache at each point in time minus the utility of the ideal, static cache configuration.
        """
        ideal_static_cache = self.gen_ideal_cache()
        ideal_static_util = sum([self.get_utility_with_state(req, ideal_static_cache) for req in self.req_hist])
        actual_util = sum(self.utility_hist)
        return actual_util - ideal_static_util
