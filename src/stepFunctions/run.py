from driver_master import Driver
import numpy as np
import torch
import sys
import multiprocessing
import logging
import glob
from PIL import Image
from time import sleep
from timeit import default_timer


# failures = 0

def worker(img_name, file_name):
    global failures
    img = Image.open(img_name)
    img = img.resize((224, 224))
    data = np.asarray(img)
    img = torch.tensor(data, dtype=torch.float).view(-1, 3, 224, 224)
    log_name = "logs/" + img_name.split(".")[0] + ".log"
    print(img_name + " processing started ")

    d = Driver("serFer.conf", img, img_name.split('.')[0], log_name, csv_file_name)
    duration = d.run()
    print(img_name + " took " + str(duration) + " seconds.")
    # if (duration is None):
    #     return_dict["failure_count"]+=1
    # print("Current failure count = {}".format(return_dict["failure_count"]))
    # return_dict["total_images"]+=1
    # print("Current number of images processed = {}".format(return_dict["total_images"]))



if __name__ == "__main__":
    print(sys.argv)
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # return_dict["failure_count"] = 0
    # return_dict["total_images"] = 0
    dir_name = sys.argv[1]+"*"
    num_files = int(sys.argv[2])
    files = [f for f in glob.glob(dir_name)]
    print(files)
    jobs = []
    count = 0
    csv_file_name = "CSV_logs" + str(default_timer()) +"_"+str(sys.argv[2])+"total_time"+ ".log"
    start_time = default_timer()
    for f in files:
        p = multiprocessing.Process(target=worker, args=(f,csv_file_name))
        sleep(0.8)
        jobs.append(p)
        p.start()
        count += 1
        if count == num_files:
            break

    # print("Total number of failures = {}".format(return_dict["failure_count"]))
    # print("Total number of images processed = {}".format(return_dict["total_images"]))
    print("Total Time taken to process = {}".format(default_timer()-start_time))
    time_taken = default_timer()-start_time
    with open(csv_file_name,"a") as f:
        f.write("Total time taken to process  = {}".format(time_taken))



