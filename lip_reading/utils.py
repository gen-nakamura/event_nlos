import time
import os
import argparse
import logging
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_LIST = []

def build_log(args):
    if not args.test:
        if not args.weights:
            timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
            log_dir = os.path.join(BASE_DIR, 'log', timestamp)
            if args.log_dir is not None:
                log_dir = os.path.join(BASE_DIR, 'log', args.log_dir)
            DIR_LIST.append(log_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = os.path.join(BASE_DIR, 'log', args.weights)
            DIR_LIST.append(log_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'train_log.txt')
        writer = SummaryWriter(os.path.join(log_dir, 'record'))
    else:
        log_file = os.path.join(BASE_DIR, 'log', args.weights, 'test_log.txt')
        writer = None
        log_dir = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    if not args.test:
        s = '-' * 15 + "Start to train" + '-' * 15 + '\n'
        for k, v in args.__dict__.items():
            s += '\t' + k + '\t' + str(v) + '\n'
    else:
        s = '-' * 15 + "Start to test" + '-' * 15 + '\n'
        for k in ['weights']:
            s += '\t' + k + '\t' + str(args.__dict__[k]) + '\n'
    logger.info(s)

    return logger, writer, log_dir

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader =  DataLoader(dataset,
                         batch_size = batch_size,
                         num_workers = num_workers,
                         shuffle = shuffle,
                         drop_last = False,
                         pin_memory=True)
    return loader