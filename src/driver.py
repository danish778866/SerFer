import boto3
import numpy as np
import torch
import redis
import pickle
from timeit import default_timer

# INPUT AND OUTPUT SIZES
input_channels = 3
W = 224  #width of input
H = 224  #height of input

split_input_sizes = [(127, 127), (8, 8), (3,3)]
split_output_sizes = [(7,7), (3,3), (10,10)]
overlap_sizes = [(1,1), (0,0), (0,0)]

lambda_client = boto3.client('lambda')

def split_image(img, inp_size):
    '''
    img - 4d tensor input of shape N x C x H x W
    
    '''
    coord = {}
    H = img.shape[2]
    W  = img.shape[3]
    h_2 = H // 2
    w_2  = W // 2
    
    coord['tl'] = [0 , inp_size[0], 0, inp_size[1]]
    coord['tr'] = [0, inp_size[0], (W-inp_size[1]), W]
    coord['bl'] = [(H-inp_size[1]), H,  0, (inp_size[1])]
    coord['br'] = [(H-inp_size[1]), H,  (W-inp_size[1]), W]
    #print(coord)
    var = coord['tl']
    #print(var)
    #img_tl_temp = torch.tensor(img[(var[0]):(var[1]), (var[2]):(var[3])], dtype = torch.float)
    img_tl = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
    print(img_tl.shape)
    #img_tl = img_tl_temp.view(-1, 1, h_2+delta, w_2+delta)
    
    var = coord['tr']
    img_tr = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
    print(img_tr.shape)
    
    var = coord['bl']
    img_bl = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
    print(img_bl.shape)
    
    var = coord['br']
    img_br = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
    print(img_br.shape)
    
    split_imgs = [[img_tl, img_tr], [img_bl, img_br]]
    return split_imgs


def merge_imgs(split_imgs, overlap):
    h = sum([s[0].shape[2] for s in split_imgs])
    w = sum([s.shape[3] for s in split_imgs[0]])
    c = split_imgs[0][0].shape[1]
    print("merged size without overlap : ", h, "  ",w, " ", c)
    print("merged size : ", h, "  ",w," ", c)
    img = torch.empty(1,c,h, w, dtype = torch.float)
    height = 0
    
    for i, split in enumerate(split_imgs):
        width = 0
        for j, s in enumerate(split):
            height_end = height + s.shape[2]
            width_end = width + s.shape[3]
            img[:, :, height:height_end, width:width_end] = s
            width += s.shape[3]    
        height += split[0].shape[2]
    print("Merged: ", img.shape)
    return img  


def all_keys_generated(lambda_keys, r):
    merge = True
    intermediate_values = []
    #print(lambda_keys)
    for idx, key in enumerate(lambda_keys):
        if idx == 0:
            current_values = []
        elif idx == 2:
            intermediate_values.append(current_values)
            current_values = []
        value = r.get(key)
        if value == None:
            merge = False
            #print("Checking for key " + key)
            break
        else:
            current_values.append(value)
    if merge:
        intermediate_values.append(current_values)
    return merge, intermediate_values

def deserialize_to_tensor(intermediate_values):
    for i, inter in enumerate(intermediate_values):
        for j, val in enumerate(inter):
            intermediate_values[i][j] = pickle.loads(intermediate_values[i][j])
            print("Inter: ", intermediate_values[i][j].shape, " ", i, " ", j)
    return intermediate_values

def main():
    start_time = default_timer()
    my_img = np.random.randn(3, 224,224)
    img = torch.tensor(my_img, dtype = torch.float).view(-1, 3, 224,224)
    fn_names = ["alex0", "alex1", "alex2"]
    driver_suffix = ".some"
    image_name = "image"
    splits = ["tl", "tr", "bl", "br"]
    print("Started...")
    r = redis.StrictRedis(host='serfer-redis.me7jbk.ng.0001.use2.cache.amazonaws.com', port=6379, db=0)
    for idx, fn_name in enumerate(fn_names[0:2]):
        print("iter : ", idx)
        s_imgs = split_image(img, split_input_sizes[idx])
        k = 0
        lambda_keys = []
        for i, split in enumerate(s_imgs):
            for j, s in enumerate(split):
                print("Going for split " + str(j))
                key_name = image_name + splits[k] + str(idx) + driver_suffix
                lambda_key_name = image_name + splits[k] + str(idx + 1)
                k = k + 1
                lambda_keys.append(lambda_key_name)
                value = pickle.dumps(s)
                r.set(key_name, value)
                payload="{\"key\": \"" + key_name + "\"}"
                print(payload)
                response = lambda_client.invoke(
                            FunctionName=fn_name,
                            InvocationType="Event",
                            Payload=payload
                        )
                print("Done split " + str(j))
        print("Done iter : ", idx)
        merge, intermediate_values = all_keys_generated(lambda_keys, r)
        while not merge:
            merge, intermediate_values = all_keys_generated(lambda_keys, r)
        # What is overlap
        intermediate_values = deserialize_to_tensor(intermediate_values)
        img = merge_imgs(intermediate_values, overlap_sizes[idx])
    key_name = image_name + "3" + driver_suffix
    lambda_key_name = image_name + "4"
    value = pickle.dumps(img)
    r.set(key_name, value)
    payload="{\"key\": \"" + key_name + "\"}"
    fn_name = "alex2"
    response = lambda_client.invoke(
               FunctionName=fn_name,
               InvocationType="Event",
               Payload=payload
           )
    f_out = r.get(lambda_key_name)
    while f_out == None:
        f_out = r.get(lambda_key_name)
    f_out_tensor = pickle.loads(f_out)
    print(f_out_tensor.shape)
    duration = default_timer() - start_time
    print("Time: ", duration)

if __name__ == "__main__":
    main()
