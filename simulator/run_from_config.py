from viz.benchmark_bip import BipartiteBenchmark

import logging

logging.getLogger().setLevel(logging.INFO)

BipartiteBenchmark("./config_4.json").run()
