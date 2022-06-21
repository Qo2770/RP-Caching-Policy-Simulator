# Credit to https://github.com/neu-spiral/OnlineCache/blob/main/tweak_traces.ipynb
#
# Y. Li, T. Si Salem, G. Neglia, and S. Ioannidis, "Online Caching Networks with Adversarial Guarantees", ACM SIGMETRICS / IFIP PERFORMANCE 2022.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def zipf_sample(s, N):
    c = sum((1 / np.arange(1, N + 1) ** s))
    return np.arange(1, N + 1) ** (-s) / c

def to_vec(x, N):
    y = np.zeros(N)
    y[x.R] = x.t
    return y

def plot_trace(rs):
    rs = np.asarray(rs)
    df = pd.DataFrame(np.array([rs[:100_000], np.arange(rs[:100_000].size) * .01]).T, columns=['R', 't']).sample(frac=1).iloc[
         :10_000]
    fig, ax = plt.subplots(figsize=(4,4))
    plt.scatter(df.t, df.R, s=0.5, color='C0')
    ax.set(ylabel='Item ID', xlabel='Time')

    plt.show()
