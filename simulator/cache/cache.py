class Cache():

    def __init__(self, size, N, T):
        self.state = set()
        self.size = size
        self.N = N
        self.T = T
        self.t = 0

        self.hits = 0
        self.req_hist = list()

        self.name = "Cache"

    def empty_space(self):
        return self.size - len(self.state)

    def evict(self, item):
        self.state.remove(item)

    def add_item(self, item):
        if self.empty_space() > 0:
            self.state.add(item)
        else:
            raise Exception("Attempted to add item to full cache!")

    def request(self, req):
        self.req_hist.append(req)

    def batch_request(self, req):
        pass

    def get_metrics(self):
        """
            Return an object that summarizes key performance metrics.
        """
        return {
            "Hits": self.hits,
            "Request count": len(self.req_hist),
            "Hit ratio": (self.hits / len(self.req_hist))
        }
