try:
    import unzip_requirements
except ImportError:
    pass

import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import boto3
import logging
import redis
import pickle
import re
#import model_layer000
import os
# import elasticache_auto_discovery
# from pymemcache.client.hash import HashClient

# #elasticache settings
# elasticache_config_endpoint = "redis.me7jbk.cfg.use2.cache.amazonaws.com:11211"
# nodes = elasticache_auto_discovery.discover(elasticache_config_endpoint)
# nodes = map(lambda x: (x[1], int(x[2])), nodes)
# r = HashClient(nodes)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AlexNet0(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet0, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

def my_replace(match):
    match = match.group()
    return str(int(match) + 1)

def lambda_handler(event, context):
    r = redis.StrictRedis(host='redis.me7jbk.ng.0001.use2.cache.amazonaws.com', port=6379, db=0)
    logger.info(event['key'])
    data_key = event['key']
    new_data_key = re.sub('[0-9]+', my_replace, data_key)
    print(new_data_key)
    new_data_key = new_data_key.split('.')[0]
    logger.info(f'New data key: {new_data_key}')
    data = r.get(data_key)
    data_tensor = pickle.loads(data)
    alex = AlexNet0()
    print(os.listdir('/opt/python/lib/python3.6/site-packages/model_layer000/'))
    alex.load_state_dict(torch.load('/opt/python/lib/python3.6/site-packages/model_layer000/model_layer000.pt'))
    data_out = alex.forward(data_tensor)
    r.set(new_data_key, pickle.dumps(data_out))
    return "Uploaded"
