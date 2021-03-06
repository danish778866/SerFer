from driver import Driver
import numpy as np
import torch
import sys
import multiprocessing
import logging
import glob
from PIL import Image
from time import sleep
from timeit import default_timer
import threading
def worker(img_name):
    img = Image.open(img_name)
    img = img.resize((224, 224))
    data = np.asarray(img)
    img = torch.tensor(data, dtype = torch.float).view(-1, 3, 224,224)
    log_name = "logs/" + img_name.split(".")[0] + ".log"
    d = Driver("serfer.conf", img, img_name.split('.')[0], log_name)
    print("Running " + str(img_name))
    stats = d.run()
    print(img_name + " Stats=" + stats)
    
if __name__ == "__main__":
    start_time = default_timer()
    dir_name = sys.argv[1]    
    num_files = int(sys.argv[2])
    files = [f for f in glob.glob(dir_name)]
    jobs = []
    count = 0
    for f in files:
        #t = threading.Thread(target=worker, args=(f,))
        #t.start()
        p = multiprocessing.Process(target=worker, args=(f,))
        #sleep(3)
        jobs.append(p)
        p.start()
        count += 1
        if count == num_files:
            break
        #if count % 500 == 0:
            #sleep(2)


#my_img = np.random.randn(3, 224,224)
#img = torch.tensor(my_img, dtype = torch.float).view(-1, 3, 224,224)
#d = Driver("serfer.conf", img, sys.argv[1])
#d.run()



