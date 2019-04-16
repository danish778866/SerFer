import redis
import torch
import pickle
import sys

r = redis.StrictRedis(host='serfer-redis.me7jbk.ng.0001.use2.cache.amazonaws.com', port=6379, db=0)
a = r.get('imagelayer1.some')
print(pickle.loads(a).shape)
#img = torch.randn(1,3,28,28)
#r.set('imagelayer0.some', pickle.dumps(img))
