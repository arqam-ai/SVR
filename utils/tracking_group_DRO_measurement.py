#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs. 
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.


import threading
import time
import os
import numpy as np


import gpustat
import logging

import itertools

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


exitFlag = 0
GPU_MEMORY_THRESHOLD = 500 # MB?

## If we need to wait for the entire clean cluster to start, select False here

all_empty = {"ind": True}
#all_empty = {"ind": False}

def num_available_GPUs(gpus):
    
    sum_i = 0
    for i, stat in enumerate(gpus):
        if stat['memory.used'] < 100:
            sum_i += 1
    return sum_i


def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        
        if num_available_GPUs(stats.gpus) >= 4:
            all_empty["ind"] = True
            
        if not all_empty["ind"]:
            logger.info("Previous experiments not finished...")
            time.sleep(10)
            continue
        
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD and i in [1,2,3]:
                return i

        logger.info("Waiting on GPUs")
        time.sleep(10)

        
class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, (bash_command, result_name) in enumerate(self.bash_command_list):
             
            import time
                
            time.sleep(0.3)
            
            if os.path.isfile(result_name):
                print("Result already exists! {0}".format(result_name))
                continue
                
            else:
                print("Result not ready yet. Running it for a second time: {0}".format(result_name))
            
            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            
            time.sleep(30)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)


class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device[0]},{self.cuda_device[1]}'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)

        logger.info("Finishing " + self.name)      

BASH_COMMAND_LIST = []

constant = 0.1
log_dir = './logs-try-load-model/'

for reweight_suffix, reweight_command in [('', ' --reweight_groups'), ('-no-reweight', ' ')]:
#for reweight_suffix, reweight_command in [('', ' --reweight_groups')]:
    for robust_suffix in ['-no-robust', '']:
#    for robust_suffix in ['-no-robust']:
        for weight_decay_suffix in ['', '-large-regularization']:
#        for weight_decay_suffix in ['']:
            result_name = f'/scratch/yyaoqing/group_DRO/results/results{robust_suffix}{reweight_suffix}{weight_decay_suffix}.pkl'
            ckpt_folder = f'logs{robust_suffix}{reweight_suffix}{weight_decay_suffix}'

            BASH_COMMAND_LIST.append((f"python model_eval_line.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --constant {constant} --batch_size 128{reweight_command} --generalization_adjustment 0 --ckpt_folder {ckpt_folder} --log_dir {log_dir} --log_name logs{robust_suffix}{reweight_suffix}{weight_decay_suffix}.txt --result_name {result_name}", result_name))


# Create new threads
dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST[:])

# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")
