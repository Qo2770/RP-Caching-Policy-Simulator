from .regret_cache import RegretCache

import logging
from numpy.random import default_rng

class RoundingCache(RegretCache):

    def __init__(self, size, N, T, seed):
        super().__init__(size, N, T)

        self.e = default_rng(seed).random()

        self.name = "Rounding Cache"

    def online_couple_round(self, xt):
        """
            Use a randomly sampeled variable `e` to perform coupled online rounding, converting a continuous cache state `xt` to a discrete one without sacrificing adversarial performance.
        """
        cache_state = set()
        xt_partial_sum = 0
        for i in range(len(xt)):
            xt_partial_sum += xt[i]
            if xt_partial_sum >= self.e + len(cache_state):
                cache_state.add(i)

        if len(cache_state) > self.size:
            logging.critical("Rounded cache exceeds maximum size!")
            logging.critical(str(cache_state))

        return cache_state
