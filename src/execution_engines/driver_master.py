import boto3
import numpy as np
import torch
import redis
import pickle
import re
from timeit import default_timer
import storage
import pdb
import json
import time
import sys
from utils import SerferConfig
import csv


def hello():
    print ("hello")

class Driver:
    def __init__(self, config_file, query, id, logfile, file_name):
        # pdb.set_trace()
        self.config_file = config_file
        self.config = SerferConfig(config_file)
        self.storage_config = self.config.get_section('Storage')
        print(self.storage_config)
        self.driver_config = self.config.get_section('Driver')
        self.step_config = self.config.get_section('StepFunction')
        self.storage_class_dict = {"redis": "SerferRedisStorage", "memcache": "SerferMemcacheStorage"}
        self.redis_storage_class = getattr(storage, self.storage_class_dict[self.storage_config["type"]])
        self.storage = self.redis_storage_class(self.storage_config["host"], self.storage_config["port"])
        self.conn = self.storage.get_storage_handle()
        self.query = query
        self.id = id
        self.storage.set_persist(True)
        self.sfn = boto3.client('stepfunctions', region_name='us-east-2', aws_access_key_id="AKIAIKZIUT4ZMPPCKWIQ",
                    aws_secret_access_key="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")
        self.logfile = logfile
        self.file_handle = open(logfile, "w+")
        self.file = open(file_name,"a")

    def run(self):
        start_time = default_timer()
        img = self.query
        key_name = self.id
        self.storage.write_to_store(key_name, img)
        # pdb.set_trace()
        # payload = json.dumps("{\"image\":"+key_name+"}")
        payload = {"image":key_name}
        sm_arn = self.step_config["fn_role"]
        response = self.sfn.start_execution(
            stateMachineArn=sm_arn,
            name="exec-1" + str(default_timer()),
            input=json.dumps(payload)
        )

        while (True):
            try:
                response = self.sfn.describe_execution(
                    executionArn=response['executionArn']
                )
                # print(response['status'])
                time.sleep(2)
                if response['status'] in ['SUCCEEDED', 'FAILED']:
                    break
            except:
                print("Throttling error, REtrying")
                continue
        try:
            assert (response['status'] == 'SUCCEEDED')
        except:
            print ("Image = {} failed".format(self.id))
            return None
        duration = default_timer() - start_time
        # self.file_handle.write("Time: " + str(duration))
        # row = [self.id, duration]
        # writer = csv.writer(self.csv_file)
        # writer.writerow(row)
        self.file.write(str(duration)+"\n")
        self.file.close()
        return duration

if __name__ == "__main__":
    hello()
