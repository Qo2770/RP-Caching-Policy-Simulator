from trace import Trace

from numpy.lib.function_base import disp
from .util import *

import numpy as np

class SlidingPopularityTrace(Trace):

    def __init__(self, N, T, roll_amount=5, parts=4, distribution=None, seed=None):
        super().__init__()
        self.name = "sliding popularity"
        
        # FIXME: T is halved for some reason
        T = (2 * T)

        if distribution is None:
            distribution = zipf_sample(0.6, N)

        window = T // parts
        final_window = (T - (window * (parts))) + window

        rng = np.random.default_rng(seed)
        
        # Generate a shifting pattern of requests
        self.reqs = rng.choice(N, window, p=distribution)
        for _ in range(parts-3):
            distribution = np.roll(distribution, roll_amount)
            self.reqs = np.hstack((self.reqs, rng.choice(N, parts, p=distribution)))
            
        distribution = np.roll(distribution, roll_amount)
        # FIXME: T is off by 4
        self.reqs = np.hstack((self.reqs, rng.choice(N, final_window, p=distribution)))[:(T // 2)]

    def has_next(self):
        return len(self.reqs) > 0

    def next(self):
        out = self.reqs[0] 
        self.reqs = self.reqs[1:]
        return out
