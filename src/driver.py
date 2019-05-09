import boto3
import numpy as np
import torch
import redis
import pickle
import re
from timeit import default_timer
import storage
from utils import SerferConfig

class Driver:
    def __init__(self, config_file, query, id, logfile):
        self.config_file = config_file
        self.config = SerferConfig(config_file)
        self.storage_config = self.config.get_section('Storage')
        self.driver_config = self.config.get_section('Driver')
        self.lambda_config = self.config.get_section('Lambda')
        self.storage_class_dict = {"redis": "SerferRedisStorage", "memcache": "SerferMemcacheStorage"}
        self.redis_storage_class = getattr(storage, self.storage_class_dict[self.storage_config["type"]])
        print(self.storage_config["mode"])
        self.storage = self.redis_storage_class(self.storage_config["host"], self.storage_config["port"], self.storage_config["mode"])
        self.conn = self.storage.get_storage_handle()
        self.query = query
        self.id = id
        self.storage.set_persist(True)
        self.lambda_client = boto3.client('lambda')
        self.logfile = logfile
        self.file_handle = open(logfile, "w")
        self.barrier_num = 0
        self.poll_time = 0
        self.layer_times = []
        self.store_pull_time = 0
        self.store_push_time = 0
        self.merge_time = 0
        self.split_time = 0
        self.current_layer = -1

    def write_microbenchmarks(self):
        self.file_handle.write("Poll Time: " + str(self.poll_time))
        self.file_handle.write("\n")
        self.file_handle.write("Layer Times: " + str(self.layer_times))
        self.file_handle.write("\n")
        self.file_handle.write("Store Push Time: " + str(self.store_push_time))
        self.file_handle.write("\n")
        self.file_handle.write("Store Pull Time: " + str(self.store_pull_time))
        self.file_handle.write("\n")
        self.file_handle.write("Merge Time: " + str(self.merge_time))
        self.file_handle.write("\n")
        self.file_handle.write("Split Time: " + str(self.split_time))
        self.file_handle.write("\n")

    def update_microbenchmarks(self, microbenchmark, time_taken):
        if microbenchmark == "poll":
            self.poll_time += time_taken
        elif microbenchmark == "get":
            self.store_pull_time += time_taken
        elif microbenchmark == "set":
            self.store_push_time += time_taken
        elif microbenchmark == "layer":
            if len(self.layer_times) == self.current_layer:
                self.layer_times.append(0)
            self.layer_times[self.current_layer] += time_taken
        elif microbenchmark == "merge":
            self.merge_time += time_taken
        elif microbenchmark == "split":
            self.split_time += time_taken


    def barrier(self, lambda_keys):
        self.storage.purge_persisted_values()
        wait = False
        self.barrier_num = self.barrier_num + 1
        for key in lambda_keys:
            if not self.storage.check_if_exists(key):
                if self.barrier_num == 100:
                    self.file_handle.write("Waiting for key " + str(key))
                    self.file_handle.write("\n")
                    self.barrier_num = 0
                wait = True
                break
        return wait

    def merge_imgs(self, split_imgs, overlap):
        h = sum([s[0].shape[2] for s in split_imgs])
        w = sum([s.shape[3] for s in split_imgs[0]])
        c = split_imgs[0][0].shape[1]
        self.file_handle.write("merged size without overlap : " + str(h) + " " + str(w) + " " + str(c))
        self.file_handle.write("\n")
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
        self.file_handle.write("Merged: " + str(img.shape))
        self.file_handle.write("\n")
        return img

    def split_image(self, img, inp_size):
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
        self.file_handle.write(str(img_tl.shape))
        self.file_handle.write("\n")
        #img_tl = img_tl_temp.view(-1, 1, h_2+delta, w_2+delta)
        
        var = coord['tr']
        img_tr = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
        self.file_handle.write(str(img_tr.shape))
        self.file_handle.write("\n")
        
        var = coord['bl']
        img_bl = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
        self.file_handle.write(str(img_bl.shape))
        self.file_handle.write("\n")
        
        var = coord['br']
        img_br = img[:, : , (var[0]):(var[1]), (var[2]):(var[3])]
        self.file_handle.write(str(img_br.shape))
        self.file_handle.write("\n")
        
        split_imgs = [[img_tl, img_tr], [img_bl, img_br]]
        return split_imgs

    def convert_to_int(self, data):
        for i, d in enumerate(data):
            size_list = d.split(",")
            size_list[0] = int(size_list[0])
            size_list[1] = int(size_list[1])
            data[i] = size_list

    def my_replace(self, match):
        match = match.group()
        return str(int(match) + 1)

    def run(self):
        start_time = default_timer()
        self.current_time = start_time
        img = self.query
        fn_names = self.lambda_config["fn_names"].split(",")
        driver_suffix = self.driver_config["suffix"]
        image_name = self.id
        splits = self.driver_config["splits"].split(",")
        split_input_sizes = self.driver_config["split_input_sizes"].split("|")
        self.convert_to_int(split_input_sizes)
        overlap_sizes = self.driver_config["overlap_sizes"].split("|")
        self.convert_to_int(overlap_sizes)
        self.storage.set_group_by(2)
        for idx, fn_name in enumerate(fn_names[0:2]):
            self.current_layer += 1
            micro_start_time = default_timer()
            s_imgs = self.split_image(img, split_input_sizes[idx])
            micro_end_time = default_timer()
            self.update_microbenchmarks("split", (micro_end_time - micro_start_time))
            k = 0
            lambda_keys = []
            start_layer_time = default_timer()
            for i, split in enumerate(s_imgs):
                for j, s in enumerate(split):
                    key_name = image_name + splits[k] +  str(idx) +  driver_suffix
                    lambda_key_name = re.sub('[0-9]+', self.my_replace, key_name)
                    lambda_key_name = lambda_key_name.split('.')[0]
                    k = k + 1
                    lambda_keys.append(lambda_key_name)
                    micro_start_time = default_timer()
                    self.storage.write_to_store(key_name, s)
                    micro_end_time = default_timer()
                    self.update_microbenchmarks("set", (micro_end_time - micro_start_time))
                    payload="{\"key\": \"" + key_name + "\"}"
                    response = self.lambda_client.invoke(
                                FunctionName=fn_name,
                                InvocationType="Event",
                                Payload=payload
                            )
            micro_start_time = default_timer()
            while self.barrier(lambda_keys):
                continue
            micro_end_time = default_timer()
            self.update_microbenchmarks("poll", (micro_end_time - micro_start_time))
            end_layer_time = default_timer()
            self.update_microbenchmarks("layer", (micro_end_time - micro_start_time))
            intermediate_values = self.storage.get_persisted_values()
            micro_start_time = default_timer()
            img = self.merge_imgs(intermediate_values, overlap_sizes[idx])
            micro_end_time = default_timer()
            self.update_microbenchmarks("merge", (micro_end_time - micro_start_time))
        self.current_layer += 1
        key_name = image_name +  "3" +  driver_suffix
        lambda_key_name = re.sub('[0-9]+', self.my_replace, key_name)
        lambda_key_name = lambda_key_name.split('.')[0]
        micro_start_time = default_timer()
        self.storage.write_to_store(key_name, img)
        micro_end_time = default_timer()
        self.update_microbenchmarks("set", (micro_end_time - micro_start_time))
        payload="{\"key\": \"" + key_name + "\"}"
        fn_name = "alex2"
        start_layer_time = default_timer()
        response = self.lambda_client.invoke(
                   FunctionName=fn_name,
                   InvocationType="Event",
                   Payload=payload
               )
        self.storage.set_group_by(1)
        micro_start_time = default_timer()
        while self.barrier([lambda_key_name]):
            continue
        micro_end_time = default_timer()
        self.update_microbenchmarks("poll", (micro_end_time - micro_start_time))
        end_layer_time = default_timer()
        self.update_microbenchmarks("layer", (micro_end_time - micro_start_time))
        f_out = self.storage.get_persisted_values()[0][0]
        self.file_handle.write(str(f_out.shape))
        self.file_handle.write("\n")
        duration = default_timer() - start_time
        self.write_microbenchmarks()
        self.file_handle.write("Time: " + str(duration))
        self.file_handle.write("\n")
        self.file_handle.close()
        return duration
