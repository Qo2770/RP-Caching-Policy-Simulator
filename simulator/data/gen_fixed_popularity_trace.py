from trace import Trace
from .util import *

import numpy as np

class FixedPopularityTrace(Trace):

    def __init__(self, N, T, distribution=None, seed=None):
        super().__init__()
        self.name = "fixed popularity"

        if distribution is None:
            distribution = zipf_sample(0.6, N)

        rng = np.random.default_rng(seed)
        self.reqs = rng.choice(N, T, p=distribution)

    def has_next(self):
        return len(self.reqs) > 0

    def next(self):
        out = self.reqs[0] 
        self.reqs = self.reqs[1:]
        return out
