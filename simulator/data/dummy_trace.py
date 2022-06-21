from trace import Trace

class DummyTrace(Trace):

    def __init__(self, N, T):
        super().__init__()
        self.name = "dummy"

        self.T = T
        self.t = 0
        self.N = N
        self.reqs = [0, 5, 2, 5, 7, 3, 9, 10, 4, 2]

    def has_next(self):
        return len(self.reqs) > 0

    def next(self):
        return self.reqs.pop(0)
