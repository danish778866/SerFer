import scipy.io as sio
import numpy as np
batch_sizes = [1,8,16,32,64,128,256,512,1000]
for batch_size in batch_sizes:
    file_name = "batch" + str(batch_size) + ".mat"
    new_name = "batch" + str(batch_size) + ".log"
    latencies = sio.loadmat(file_name)['latencies']
    np.savetxt(new_name, latencies, delimiter='\n')
