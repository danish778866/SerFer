from driver import Driver
import numpy as np
import torch
import sys
import multiprocessing
import logging
import glob
from PIL import Image
from time import sleep
def worker(img_name):
    img = Image.open(img_name)
    img = img.resize((224, 224))
    data = np.asarray(img)
    img = torch.tensor(data, dtype = torch.float).view(-1, 3, 224,224)
    log_name = "logs/" + img_name.split(".")[0] + ".log"
    d = Driver("serfer.conf", img, img_name.split('.')[0], log_name)
    duration = d.run()
    print(img_name + " took " + str(duration) + " seconds.")
    
if __name__ == "__main__":
    dir_name = sys.argv[1]    
    files = [f for f in glob.glob(dir_name)]
    print(files)
    jobs = []
    for f in files:
        p = multiprocessing.Process(target=worker, args=(f,))
        #sleep(3)
        jobs.append(p)
        p.start()


#my_img = np.random.randn(3, 224,224)
#img = torch.tensor(my_img, dtype = torch.float).view(-1, 3, 224,224)
#d = Driver("serfer.conf", img, sys.argv[1])
#d.run()



