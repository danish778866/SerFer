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
# import elasticache_auto_discovery
# from pymemcache.client.hash import HashClient

# #elasticache settings
# elasticache_config_endpoint = "redis.me7jbk.cfg.use2.cache.amazonaws.com:11211"
# nodes = elasticache_auto_discovery.discover(elasticache_config_endpoint)
# nodes = map(lambda x: (x[1], int(x[2])), nodes)
# r = HashClient(nodes)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AlexNet2(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet2, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# class AlexNet2(nn.Module):

#     def __init__(self, num_classes=1000):
#         super(AlexNet2, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.features = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 1000, kernel_size=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=6, stride=1),
#         )

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.features(x)
#         x = x.view(-1,1000)
#         return x

def my_replace(match):
    match = match.group()
    return str(int(match) + 1)

def lambda_handler(event, context):
    r = redis.StrictRedis(host='redis.me7jbk.ng.0001.use2.cache.amazonaws.com', port=6379, db=0)
    logger.info(event['key'])
    data_key = event['key']
    new_data_key = re.sub('[0-9]+', my_replace, data_key)
    new_data_key = new_data_key.split('.')[0]
    logger.info(f'New data key: {new_data_key}')
    data = r.get(data_key)
    logger.info(f'Done: {new_data_key}')
    data_tensor = pickle.loads(data)
    logger.info("At least")
    alex = AlexNet2()
    #weights_pickle = r.get('alex2')
    #weights = pickle.loads(weights_pickle)
    #alex.load_state_dict(weights)
    alex.load_state_dict(torch.load('/opt/python/lib/python3.6/site-packages/model_layer002/model_layer002.pt'))
    logger.info("Wow")
    data_out = alex.forward(data_tensor)
    logger.info("I am here")
    r.set(new_data_key, pickle.dumps(data_out))
    return "Uploaded"

