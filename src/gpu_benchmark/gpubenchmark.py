from __future__ import print_function
import numpy as np
import torch
import torchvision
import torchvision.models as models
import time
import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.utils import check_integrity, download_url
import scipy.io as sio
import sys

def getdataloader(dir, batchsize = 1):
    data_transforms = {
            'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
            'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}    
    return dataloaders['train']

def getmodel():
    return models.alexnet(pretrained = True)

def getdevice():
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device : ', device)
    return device

def infer(batchsize, filename):
    #batchsize = int(sys.argv[1])
    device = getdevice()
    dataloader = getdataloader('./hymenoptera_data', batchsize)
    model = models.alexnet(pretrained = True).to(device)
    model.eval()
    latencies = []
    count = 0
    ttic = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
      tic = time.time() 
      X, y = sample_batched[0].to(device), sample_batched[0].to(device)
      
      y_scores = model(X)
      toc = time.time()
      latency = toc - tic
      latencies.append(latency)
      count += batchsize
      if count > 16000:
        break
    ttoc = time.time()
    totaltime = ttoc - ttic
    print('batchsize:', batchsize)
    print('total time : ', totaltime)
    print('count :', count)
    throughput = count / totaltime
    print('throughput:', throughput)
    print(latencies)
    mydict =  {'latencies':latencies, 'throughput':throughput, 'totaltime':totaltime, 'count':count}
    #sio.savemat(filename, mydict)
    return mydict

def main():
    for i in [1, 8, 16, 32, 64, 128, 256, 512, 1000]:
        mydict = infer(i, 'batch'+str(i)+'.mat')
        tdict = {}
        tdict[i] = mydict['throughput']
        sio.savemat('batch'+str(i)+'.mat', mydict)
    sio.savemat('throughput.mat', tdict)
    print(tdict)

main()
