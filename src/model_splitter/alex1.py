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
    import collections
    # import elasticache_auto_discovery
    # from pymemcache.client.hash import HashClient
    
    # #elasticache settings
    # elasticache_config_endpoint = "redis.me7jbk.cfg.use2.cache.amazonaws.com:11211"
    # nodes = elasticache_auto_discovery.discover(elasticache_config_endpoint)
    # nodes = map(lambda x: (x[1], int(x[2])), nodes)
    # r = HashClient(nodes)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    class AlexNet1(nn.Module):
    
        def __init__(self, num_classes=1000):
            super(AlexNet1, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    
        def forward(self, x):
            x = self.features(x)
            #x = self.avgpool(x)
            return x
    
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
        data_tensor = pickle.loads(data)
        alex = AlexNet1()
        w_keys = list(alex.state_dict().keys())
        print(alex.state_dict().keys())
        #alex.load_state_dict(torch.load('/opt/python/lib/python3.6/site-packages/model_layer001/model_layer001.pt'))
        wmodel = torch.load('/opt/python/lib/python3.6/site-packages/model_layer001/model_layer001.pt')
        my_dict = collections.OrderedDict()
        dummykeys = list(wmodel.keys())
        print(dummykeys)
        for i,k in enumerate(w_keys):
            my_dict[k] = wmodel[dummykeys[i]]
        alex.load_state_dict(my_dict)
        data_out = alex.forward(data_tensor)
        r.set(new_data_key, pickle.dumps(data_out))
        return "Uploaded"

