from .cache import Cache

class LRU(Cache):

    def __init__(self, size, N, T):
        super().__init__(size, N, T)
        
        # Stores the age of each element in the cache
        self.age_bits = {}

        self.name = "LRU"

    def request(self, req):
        """
            Input a request to the cache. If there is no room left, evit the least recently used element.
            Return True if a cache hit is achieved.
        """
        super().request(req)

        # Update the age of each element in the cache
        self.age_bits = {i: (age + 1) for i, age in self.age_bits.items()}

        if req in self.state:
            # Cache hit, no change to state necessary
            self.hits += 1
            self.age_bits[req] = 0
            return True

        if self.empty_space() > 0:
            # Cache not full, add the requested element
            self.add_item(req)
            self.age_bits[req] = 0
            return False
        
        # Find the least recently updated (highest age) and evict
        lru_item = max(self.age_bits, key=self.age_bits.get)
        self.age_bits.pop(lru_item)
        self.evict(lru_item)

        # Add the requested element to cache
        self.add_item(req)
        self.age_bits[req] = 0

        return False

