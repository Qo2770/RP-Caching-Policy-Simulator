import datetime
import logging
import json
from time import perf_counter
import os

from viz.hit_ratio_vs_time import optimal_caching_over_time
from viz.hit_ratio_vs_time import plot_data_series_and_save
from viz.hit_ratio_vs_time import save_series_as_csv

from tqdm import tqdm

OPTIMAL_CACHE_NAME = "Optimal Static Cache"
CSV_FILE_NAME = "hit_ratio_vs_time.csv"
FIGURE_FILE_NAME = "hit_ratio_vs_time.png"
PARAM_FILE_NAME = "params.json"

class Benchmark():

    def __init__(self, trace, caches, seed, save_path='./results/', subfolder_name=None, generate_optimal=True, save_csv=True, save_graphs=True):
        """
            Init.
        """
        # Create a list of the caching algorithm names to separate results
        self.cols = list()
        for c in caches:
            self.cols.append(c.name)
        
        # Append the name of the optimal policy if one should be benchmarked against 
        if generate_optimal:
            self.cols.append(OPTIMAL_CACHE_NAME)

        self.caches = caches
        self.data = trace
        self.seed = seed
        
        self.generate_optimal = generate_optimal
        self.save_csv = save_csv
        self.save_graphs = save_graphs

        self.req_hist = list()

        # If no custom subfolder name is set, use the current ISO datetime yyyy-mm-dd-hh:mm:ss
        if subfolder_name is None:
            date = datetime.datetime.now().isoformat(' ').split(' ')
            subfolder_name = date[0] + "-" + date[1].split('.')[0].replace(':', '.') + '/'
        
        # Add a '/' to the end of folder paths if not present
        if save_path[-1] != '/':
            save_path += '/'
        if subfolder_name[-1] != '/':
            subfolder_name += '/'

        # Create a subfolder to contain all results
        os.mkdir(save_path + subfolder_name)

        self.save_folder = subfolder_name
        self.save_path = save_path


    def run_benchmark(self):
        """
            Run the benchmark and generate results.
        """
        # Save a history of hit ratios for each cache under benchmark
        hit_ratios = dict()
        for c in self.caches:
            hit_ratios[c.name] = list()
        
        # Main loop, using tqdm to visualize progress
        for _ in tqdm(range(len(self.data.reqs))):
            
            # The loop should not go beyond T
            if not (self.data.has_next()):
                logging.error("The loop exceeded trace length!")
                break
            
            # Retrieve the request and add to the history of requests (used for generating optimal policy)
            req = self.data.next()
            self.req_hist.append(int(req))

            # Build a dict of hits and hit ratios for each cache
            res = dict()
            for cache in self.caches:
                res[cache.name] = "Hit" if cache.request(req) else "Miss"
                hit_ratios[cache.name].append(cache.get_metrics()["Hit ratio"])

            logging.debug("Requested " + str(req) + ": " + str(res))
        
        # The loop should consume the entire trace
        if self.data.has_next():
            logging.warn("The loop did not fully consume the trace!")
        
        # Generate and benchmark optimal static caching policy
        if self.generate_optimal:
            hit_ratios[OPTIMAL_CACHE_NAME] = optimal_caching_over_time(self.req_hist, self.caches[0].size)

        for c in self.caches:
            logging.info(str(c.name) + ": " + str(c.get_metrics()))

        if self.save_csv:
            logging.info("Saving benchmark CSV in " + self.save_folder + CSV_FILE_NAME)
            save_series_as_csv(hit_ratios, self.save_folder + CSV_FILE_NAME, file_path=self.save_path)
            logging.info("CSV saved!")

        if self.save_graphs:
            logging.info("Graphing metrics and saving in " + self.save_folder + FIGURE_FILE_NAME)
            plot_data_series_and_save(hit_ratios, self.cols, len(self.req_hist), self.data.name, self.save_folder + FIGURE_FILE_NAME, file_path=self.save_path)
            logging.info("Graph saved!")

        logging.info("Saving seed and request history to " + self.save_folder + PARAM_FILE_NAME)
        
        # Create object to store request history and seed
        benchmark_params = dict()
        benchmark_params["seed"] = int(self.seed)
        benchmark_params["trace"] = self.req_hist
        
        # Save params to file
        params_out = json.dumps(benchmark_params)
        param_file = open(self.save_path + self.save_folder + PARAM_FILE_NAME, "w")
        param_file.write(params_out)
        param_file.close()

        logging.info("Parameters saved!")

        logging.info("=== Benchmark complete! ===")


        
