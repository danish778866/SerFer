from driver import Driver
import numpy as np
import torch
import sys

my_img = np.random.randn(3, 224,224)
img = torch.tensor(my_img, dtype = torch.float).view(-1, 3, 224,224)
d = Driver("serfer.conf", img, sys.argv[1])
d.run()



