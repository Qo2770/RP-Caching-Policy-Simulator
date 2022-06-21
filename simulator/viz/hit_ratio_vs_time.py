import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def optimal_caching_over_time(reqs, cache_size):
    """
        Return an array of hit ratio over time for the best static cache state in hindsight.
    """
    uniq, counts = np.unique(reqs, return_counts=True)
    max_reqs = np.argpartition(counts, -cache_size)[-cache_size:]
    static_state = uniq[max_reqs]

    utility_over_time = list()
    c = 0
    hits = 0
    for req in reqs:
        c += 1
        if req in static_state:
            hits += 1
        utility_over_time.append(hits / c)

    return utility_over_time

def get_one_hot_reqs(reqs, src_len, N):

    reqs_one_hot = np.zeros((src_len, N))

    for src in reqs:
        reqs_one_hot[src, :] = np.array([int(i in reqs[src]) for i in range(N)])

    return reqs_one_hot

def get_util(reqs, src_len, cache_len, cache, weights, N, cache_orders):

    reqs_enc = get_one_hot_reqs(reqs, src_len, N)

    out = 0

    for n in range(N):

        for i in range(src_len):

            for k, j in enumerate(cache_orders[i]):

                out += weights[i, j, n] * min(1, reqs_enc[i, n] * np.sum([ int(n in cache[j_i]) for j_i in cache_orders[i][:k+1] ]) )

    return out

def get_cost(reqs, src_len, cache_len, cache, weights, N, cache_orders):

    max_cost = np.amax(weights)
        
    reqs_enc = get_one_hot_reqs(reqs, src_len, N)
    cost = 0

    for i in range(src_len):
        for el in np.argwhere(reqs_enc[i, :]):
            n = el[0]            
            cost += reqs_enc[i, n] * ( max_cost - np.amax( [weights[i, j, n] * int(n in cache[j]) for j in range(cache_len)] ) )

    return cost

def benchmark_util_static_bipartite_cache(cache, weights, N, reqs, cache_orders):
    util = 0
    t = 0
    util_hist = list()

    for req in reqs:

        req_count = 0
        for i in req:
            for _ in req[i]:
                req_count += 1
        
        t += 1
        util += get_cost(req, len(req), len(cache), cache, weights, N, cache_orders) / req_count
        util_hist.append(util / t)

    return util_hist


def benchmark_static_bipartite_cache(cache, weights, reqs):
    hits = 0
    req_items = 0
    hit_ratio_hist = list()
    for req in reqs:
        for src_idx, src_req in enumerate(req):
            for r in req[src_req]:
                hits += min(1.0, np.sum(
                    [
                        weights[src_req, j, r] * 
                            int(r in cache[j]) 
                        for j, _ in enumerate(cache)
                    ]
                ))
                req_items += 1
        hit_ratio_hist.append(hits / req_items)

    return hit_ratio_hist

def save_series_as_csv(x, file_name, file_path="./results/"):
    """
        Saves a dict as a CSV file.
    """
    df = pd.DataFrame(data=x)
    df.to_csv(str(file_path + file_name))

def plot_data_series_and_save(x, cols, T, trace_name, file_name, file_path="./results/"):
    """
        Plot dict with multiple cols and save.
    """
    df = pd.DataFrame(data=x)

    plt.rc('font', size=25)     

    SMALL_SIZE = 20
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 25

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    

    fig, ax = plt.subplots(figsize=[15, 8])
    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    x = np.linspace(0, T, T)

    for c in cols:
        if "optimal" in str(c).lower():
            ax.plot(x, df[c], linewidth=2.5, linestyle='--', label=c)
        else:
            ax.plot(x, df[c], linewidth=2.5, label=c)

    ax.set_xlabel('Time')  
    ax.set_ylabel('NAC')  
    #ax.set_title('Hit ratio over time on a ' + str(trace_name) + ' trace')  

    ax.legend()
    
    plt.savefig(file_path + file_name)
    plt.show()

