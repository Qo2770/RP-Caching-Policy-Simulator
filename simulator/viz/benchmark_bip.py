import json
import logging
import os

import numpy as np
from tqdm import tqdm

from data.gen_circ_adversarial_trace import CircularAdversarialTrace
from data.gen_fixed_popularity_trace import FixedPopularityTrace
from data.gen_sliding_popularity_trace import SlidingPopularityTrace

from cache.bipartite_omd_integral_cache import BipartiteIntegralOMD
from cache.bipartite_omd_cache import BipartiteOMD
from cache.bipartite_lfu import BipartiteLFU
from cache.bipartite_omd_single import BipartiteOMDNaive

from viz.hit_ratio_vs_time import benchmark_static_bipartite_cache, benchmark_util_static_bipartite_cache
from viz.hit_ratio_vs_time import plot_data_series_and_save
from viz.hit_ratio_vs_time import save_series_as_csv

OPTIMAL_CACHE_NAME = "Optimal Static Cache"
CSV_FILE_NAME = "hit_ratio_vs_time.csv"
FIGURE_FILE_NAME = "hit_ratio_vs_time.png"
PARAM_FILE_NAME = "params.json"

class BipartiteBenchmark():

    def __init__(self, config_path):
        
        with open(config_path, 'r') as config_file:
            
            config_data = json.load(config_file)

            if not ("seed" in config_data) or config_data["seed"] == "NONE":
                self.seed = np.random.default_rng().bit_generator._seed_seq.entropy
            else:
                self.seed = config_data["seed"]

            self.rng = np.random.default_rng(self.seed)

            self.save_path = config_data["save_path"]
        
            self.plot_optimal = config_data["plot_optimal"]
            self.save_csv = config_data["save_csv"]
            self.save_graph = config_data["save_graph"]
            self.batch_size = config_data["batch_size"]
            self.N = config_data["catalog_size"]
            self.T = config_data["time"] 
            
            self.sources = list()
            for src in config_data["sources"]:
                self.sources.append(self.trace_from_str(src))

            self.ks = list()
            self.weights = list()
            self.step_size = list()
            self.caches = list()
            self.batch_sizes = list()

            self.cache_orders = list()
            for _ in range(len(self.sources)):
                self.cache_orders.append(self.rng.permutation(range(len(config_data["cache_systems"][0]["sizes"]))))

            for idx, cache in enumerate(config_data["cache_systems"]):

                self.ks.append(cache["sizes"])

                weights = np.zeros((len(self.sources), len(self.ks[idx]), self.N)) 

                for link in config_data["links"]:
                    weights[link[0], link[1], :] = link[2]
                 
                step_size = None
                if "step_size" in cache:
                    step_size = cache["step_size"]
                    self.step_size.append(cache["step_size"])
                else:
                    self.step_size.append(None)

                if "batch_size" in cache:
                    self.batch_sizes.append(cache["batch_size"])
                else:
                    self.batch_sizes.append(self.batch_size)

                gen_cache = self.cache_from_str(cache["type"], self.ks[idx], weights, step_size=step_size, cache_orders=self.cache_orders)
                if "name" in cache:
                    gen_cache.name = cache["name"] 
                self.caches.append(gen_cache)

                self.weights.append(weights)

            self.params = config_data
            self.params["seed"] = self.seed

    def run(self):
        
        t = 0
        req_hist = list()
        hit_ratios = dict()

        req_buf = [dict() for _ in range(len(self.caches))]
        batch_counter = [0 for _ in range(len(self.caches))]
 
        logging.getLogger().setLevel(logging.WARN)

        for _ in tqdm(range(self.T)):
            
            req = dict()

            for src_idx, src in enumerate(self.sources):
                nxt = src.next()
                for req_cache in req_buf:
                    if src_idx in req_cache:
                        req_cache[src_idx].append(nxt) 
                    else:
                        req_cache[src_idx] = [nxt]

                if src_idx in req:
                    req[src_idx].append(nxt) 
                else:
                    req[src_idx] = [nxt]


            for idx, cache in enumerate(self.caches):
                batch_counter[idx] += 1
                if batch_counter[idx] >= self.batch_sizes[idx] or (t+1) == self.T:
                    cache.request(req_buf[idx])
                    batch_counter[idx] = 0
                    req_buf[idx] = dict()
            req_hist.append(req)

            t += 1

        logging.getLogger().setLevel(logging.INFO)

        cols = list()

        for idx, cache in enumerate(self.caches):
            util_hist = list()
            for util in cache.util_hist:
                for _ in range(self.batch_sizes[idx]):
                    util_hist.append(util)
            print(len(util_hist[:self.T]))
            hit_ratios[cache.name] = util_hist[:self.T]
            cols.append(cache.name)

        os.mkdir("./results/" + self.save_path)

        if self.plot_optimal:
            opt_cache = self.caches[0].generate_optimal_state()
            optimal_hit_ratio = benchmark_util_static_bipartite_cache(opt_cache, self.weights[0], self.N, req_hist, self.cache_orders) 
            hit_ratios[OPTIMAL_CACHE_NAME] = optimal_hit_ratio
            cols.append(OPTIMAL_CACHE_NAME)

        if self.save_csv:
            logging.info("Saving benchmark CSV in " + self.save_path + CSV_FILE_NAME)
            save_series_as_csv(hit_ratios, self.save_path + "/" + CSV_FILE_NAME, file_path="./results/")
            logging.info("CSV saved!")

        if self.save_graph:
            logging.info("Graphing metrics and saving in " + self.save_path + FIGURE_FILE_NAME)
            plot_data_series_and_save(hit_ratios, cols, len(req_hist), ", ".join([src.name for src in self.sources]), self.save_path + "/" + FIGURE_FILE_NAME, file_path="./results/")
            logging.info("Graph saved!")

        self.save_params("./results/" + self.save_path)
            
       
    def save_params(self, path):
        """
            Save params to file
        """ 
        params_out = json.dumps(self.params)
        param_file = open(path + "/" + PARAM_FILE_NAME, "w")
        param_file.write(params_out)
        param_file.close() 

    def trace_from_str(self, trace_str):

        if trace_str == "Fixed":
            return FixedPopularityTrace(self.N, self.T * self.batch_size, seed=self.rng)
        elif trace_str == "Circular":
            return CircularAdversarialTrace(self.N, self.T * self.batch_size)
        elif trace_str == "Sliding":
            return SlidingPopularityTrace(self.N, self.T * self.batch_size, seed=self.rng)

        raise NotImplemented("Cannot load trace from file")

    def cache_from_str(self, cache_str, sizes, weights, step_size=None, cache_orders=None):

        if cache_str == "Integral Bipartite":
            return BipartiteIntegralOMD(self.sources, sizes, self.N, self.T, weights=weights, cache_orders=cache_orders, step_size=step_size, seed=self.rng)
        elif cache_str == "Fractional Bipartite":
            return BipartiteOMD(sizes, self.sources, weights, self.N, self.T, self.seed)
        elif cache_str == "LFU Bipartite":
            return BipartiteLFU(self.sources, sizes, self.N, self.T, weights=weights)
        elif cache_str == "OMD Naive Bipartite":
            return BipartiteOMDNaive(self.sources, sizes, self.N, self.T, weights=weights)
        
        raise NotImplemented("This cache type is not implemented")
