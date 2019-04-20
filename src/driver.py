import boto3
import numpy as np
import torch
import redis
import pickle
from timeit import default_timer
import storage
from utils import SerferConfig

class Driver:
    def __init__(self, config_file, query, id):
        self.config_file = config_file
        self.config = SerferConfig(config_file)
        self.storage_config = self.config.get_section('Storage')
        self.driver_config = self.config.get_section('Driver')
        self.lambda_config = self.config.get_section('Lambda')
        self.storage_class_dict = {"redis": SerferRedisStorage}
        self.redis_storage_class = getattr(storage, self.storage_class_dict[self.storage_config["type"]])
        self.storage = redis_storage_class(self.storage_config["host"], self.storage_config["port"])
        self.conn = self.storage.get_storage_handle()
        self.query = query
        self.id = id
        self.storage.persist()
        self.lambda_client = boto3.client('lambda')

    def barrier(lambda_keys):
        self.storage.purge_persisted_values()
        wait = False
        for key in lambda_keys:
            if not self.storage.check_if_exists(key):
                wait = True
                break
        return wait

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


    def run():
        start_time = default_timer()
        img = query
        fn_names = self.lambda_config["fn_names"].split(",")
        driver_suffix = self.driver_config["suffix"]
        image_name = self.id
        splits = self.driver_config["splits"].split(",")
        split_input_sizes = self.driver_config["split_input_sizes"].split("|")
        overlap_sizes = self.driver_config["overlap_sizes"].split("|")
        for idx, fn_name in enumerate(fn_names[0:2]):
            s_imgs = split_image(img, split_input_sizes[idx])
            k = 0
            lambda_keys = []
            for i, split in enumerate(s_imgs):
                for j, s in enumerate(split):
                    key_name = image_name + splits[k] + str(idx) + driver_suffix
                    lambda_key_name = image_name + splits[k] + str(idx + 1)
                    k = k + 1
                    lambda_keys.append(lambda_key_name)
                    self.conn.write_to_store(key_name, s)
                    payload="{\"key\": \"" + key_name + "\"}"
                    response = self.lambda_client.invoke(
                                FunctionName=fn_name,
                                InvocationType="Event",
                                Payload=payload
                            )
            while self.barrier(lambda_keys):
                continue
            intermediate_values = self.storage.get_persisted_values()
            img = merge_imgs(intermediate_values, overlap_sizes[idx])
        key_name = image_name + "3" + driver_suffix
        lambda_key_name = image_name + "4"
        self.conn.write_to_store(key_name, img)
        payload="{\"key\": \"" + key_name + "\"}"
        fn_name = "alex2"
        response = self.lambda_client.invoke(
                   FunctionName=fn_name,
                   InvocationType="Event",
                   Payload=payload
               )
        self.storage.group_by(1)
        while self.barrier([lambda_key_name]):
            continue
        f_out = self.storage.get_persisted_values()[0][0]
        print(f_out_tensor.shape)
        duration = default_timer() - start_time
        print("Time: ", duration)
