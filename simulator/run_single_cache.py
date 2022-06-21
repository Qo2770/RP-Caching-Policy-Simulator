from cache.omd_cache import OMD
from cache.oga_cache import OGA
from cache.lru_cache import LRU
from cache.lfu_cache import LFU
from data.gen_circ_adversarial_trace import CircularAdversarialTrace
from data.gen_fixed_popularity_trace import FixedPopularityTrace
from data.gen_sliding_popularity_trace import SlidingPopularityTrace
from viz.benchmark import Benchmark
from viz.hit_ratio_vs_time import *

import logging
from time import perf_counter

import numpy as np

logging.getLogger().setLevel(logging.INFO)

CACHE_SIZE = 20
CATALOG_SIZE = 100
T = 50

SEED = None

# If seed is not set, let numpy geenrate one
if SEED is None:
    SEED = np.random.default_rng().bit_generator._seed_seq.entropy

logging.info("Using seed " + str(SEED))

step_sizes = [i / 10 for i in range(11)]
caches = [OMD(CACHE_SIZE, CATALOG_SIZE, T, step_size=k, seed=SEED) for k in step_sizes]
caches.append(OMD(CACHE_SIZE, CATALOG_SIZE, T, seed=SEED))
for c in caches:
    c.name += ", Step size = " + str(c.step_size) 

data = FixedPopularityTrace(CATALOG_SIZE, T, seed=SEED)
# cache_lru = LRU(CACHE_SIZE, CATALOG_SIZE, T)
# cache_lfu = LFU(CACHE_SIZE, CATALOG_SIZE, T)
# cache_oga = OGA(CACHE_SIZE, CATALOG_SIZE, T)
# cache_omd = OMD(CACHE_SIZE, CATALOG_SIZE, T)

benchmark = Benchmark(data, caches, SEED)
benchmark.run_benchmark()
