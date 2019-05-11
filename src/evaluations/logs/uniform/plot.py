import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

def plot_cdfs(files, title, xlabel, ylabel, labels=None):
    plt.set_cmap('copper')
    style.use('seaborn-poster') #sets the size of the charts
    style.use('ggplot')
    plt.rcParams.update({'font.size': 14})
    plt.rc('legend',fontsize=15)
    plt.rc('legend',loc='lower right')
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, f in enumerate(files):
        x = np.fromfile(f, sep='\n')
        counts, bin_edges = np.histogram (x, bins=200, density=True)
        cdf = np.cumsum (counts)
        if labels is None:
            ax.plot(bin_edges[1:], cdf/cdf[-1])
        else:
            ax.plot(bin_edges[1:], cdf/cdf[-1], label=labels[idx])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()

#plot_cdfs(['merge_1000.log', 'split_1000.log'],  'Merge and Split microbenchmarks',
#         'Latency(s)', 'CDF', labels=['Merge', 'Split'])

#plot_cdfs(['poll_1000.log'], 'Polling time for Poll Driver', 'Latency(s)', 'CDF')

#plot_cdfs(['get_1000.log', 'set_1000.log'], ['Read', 'Write'],
#         'Redis Read and Write microbenchmarks', 'Latency(s)', 'CDF')

#plot_cdfs(['layer_1_1000.log', 'layer_2_1000.log', 'layer_3_1000.log'], 
#         'Microbenchmarks for Lambda Layers', 'Latency(s)', 'CDF', labels=['Lambda Layer 1', 'Lambda Layer 2', 'Lambda Layer 3'])

#plot_cdfs(['test.log', 'test_2.log', 'test_3.log', 'test_4.log'], ['1000', '500', '250', '100'],
#         'Latency CDFs for different bursts', 'Latency(s)', 'CDF')

plot_cdfs(['latency_1000_sleep1.log', 'latency_1000_sleep0.2.log', 'latency_1000_sleep0.02.log', 'test.log'], 
         'Latency for uniform traffic', 'Latency(s)', 'CDF', labels=['Sleep 1s', 'Sleep 200ms', 'Sleep 20ms', 'Burst 1000'])

def plot_throughput():
    plt.set_cmap('copper')
    style.use('seaborn-poster') #sets the size of the charts
    style.use('ggplot')
    plt.rcParams.update({'font.size': 14})
    plt.rc('legend',fontsize=15)
    plt.rc('legend',loc='lower right')
    plt.figure(1, figsize=(5, 3.5))
    x = [100, 250, 500, 1000]
    y = [78.125, 148.5, 267.86, 411]
    plt.plot(x, y)
    plt.title("Throughput vs Burst Size")
    plt.xlabel("Burst Size")
    plt.ylabel("Queries Processed (per minute)")
    plt.tight_layout()
    plt.show()

#plot_throughput()
##x = np.fromfile('test.log', sep='\n')
##y = np.fromfile('test_2.log', sep='\n')
##x2 = np.fromfile('test_3.log', sep='\n')
##x3 = np.fromfile('test_4.log', sep='\n')
###count, bins, patches = plt.hist(x, bins=200, cumulative=True, histtype='step')
##counts, bin_edges = np.histogram (x, bins=200, density=True)
##cdf = np.cumsum (counts)
##counts_1, bin_edges_1 = np.histogram (y, bins=200, density=True)
##cdf_1 = np.cumsum (counts_1)
##counts_2, bin_edges_2 = np.histogram (x2, bins=200, density=True)
##cdf_2 = np.cumsum (counts_2)
##counts_3, bin_edges_3 = np.histogram (x3, bins=200, density=True)
##cdf_3 = np.cumsum (counts_3)
###cdf = count /1000
###plt.figure(1, figsize=(5, 3.5))
##plt.xticks([0,2,4,8,12,16,20,40,80,100])
##ax.plot(bin_edges[1:], cdf/cdf[-1], label='Hello')
##ax.plot(bin_edges_1[1:], cdf_1/cdf_1[-1])
##ax.plot(bin_edges_2[1:], cdf_2/cdf_2[-1])
##ax.plot(bin_edges_3[1:], cdf_3/cdf_3[-1])
##ax.legend()
##plt.tight_layout()
##plt.show()
