from trace import Trace

from .util import plot_trace

import numpy as np

class CircularAdversarialTrace(Trace):

    def __init__(self, N, T):
        super().__init__()
        self.name = "circular adversarial"

        # Credit to https://github.com/neu-spiral/OnlineCache/blob/main/tweak_traces.ipynb
        #
        # Y. Li, T. Si Salem, G. Neglia, and S. Ioannidis, "Online Caching Networks with Adversarial Guarantees", ACM SIGMETRICS / IFIP PERFORMANCE 2022.
        #
        
        self.reqs = list()
        
        cat1 = np.arange(N // 2)
        cat2 = np.arange(N // 2, N)
        for i in range(T // 2):
            self.reqs += [cat1[i % (N // 2)]]
            self.reqs += [cat2[i % (N // 2)]]
        
    def has_next(self):
        return len(self.reqs) > 0

    def next(self):
        out = self.reqs[0] 
        self.reqs = self.reqs[1:]
        return out
