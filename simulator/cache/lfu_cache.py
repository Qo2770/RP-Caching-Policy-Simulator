from .cache import Cache

class LFU(Cache):
    
    def __init__(self, size, N, T):
        super().__init__(size, N, T)
        
        # A dict to keep track of how often each element in the cache is used
        self.freq_bits = {}

        self.name = "LFU"

    def request(self, req):
        """
            Input a request to the cache. If there is no room left, evit the least frequently used element.
            Return True if a cache hit is achieved.
        """
        super().request(req)

        
        if req in self.state:
            # Cache hit, update the frequency of the reuqested item
            self.hits += 1
            self.freq_bits[req] += 1
            return True

        if self.empty_space() > 0:
            # Cache not full, add the requested item
            self.add_item(req)
            self.freq_bits[req] = 0
            return False
        
        # Evice the item that has been used the least
        lru_item = min(self.freq_bits, key=self.freq_bits.get)
        self.freq_bits.pop(lru_item)
        self.evict(lru_item)
        
        # Add the requested item to the cache
        self.add_item(req)
        self.freq_bits[req] = 0

        return False
