#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Run as python3 -m train.train

import argparse
import copy
import logging
import torch
import torch.multiprocessing as mp
from train import arg_parser
import os
import time
import getpass
import shutil
from distutils.dir_util import copy_tree
import sys

from train import polybeast, pantheon_env, utils
from train.constants import THIRD_PARTY_ROOT

sys.path.append(THIRD_PARTY_ROOT)
logging.basicConfig(level=logging.INFO)
os.environ["OMP_NUM_THREADS"] = "1"

def get_parser():
    parser = argparse.ArgumentParser()
    polybeast_parser = parser.add_argument_group("polybeast")
    pantheon_parser = parser.add_argument_group("pantheon_env")
    arg_parser.add_polybeast_args(polybeast_parser)
    arg_parser.add_pantheon_args(pantheon_parser)
    arg_parser.add_common_args(parser)
    return parser


def init_logdirs(flags):
    # Clean run for test mode
    os.makedirs(flags.base_logdir, exist_ok=True)   
    if(flags.mode == "train"):
        flags.logdir = os.path.join(os.path.join(flags.base_logdir, flags.mode))
        checkpoint_dir = os.path.join(flags.logdir, "checkpoints")
        if os.path.exists(os.path.join(checkpoint_dir, "checkpoints")):
            assert len(os.listdir(checkpoint_dir)) == 0, "checkpoints directory not empty"
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if(flags.checkpoint == None):
            flags.checkpoint = os.path.join(checkpoint_dir, "checkpoint.tar")
    elif flags.mode == 'test':
        flags.log_dir = os.path.join(flags.base_logdir, flags.mode, flags.test_name)
    
    flags.traced_model = os.path.join(flags.base_logdir, "traced_model.pt")
    os.makedirs(flags.logdir, exist_ok=True)  
    flags.traced_model = os.path.join(flags.base_logdir, "traced_model.pt")
    if flags.mode != "train":
        assert os.path.exists(
            flags.checkpoint
        ), "Checkpoint {} missing in {} mode".format(flags.checkpoint, flags.mode)


def run_remote(flags, mode='train'):
    assert((mode == 'train' or mode == 'test') and mode == flags.mode)
    init_logdirs(flags)
    
    use_cuda = flags.disable_cuda and torch.cuda.is_available()
    device = "cuda:{}".format(flags.gpu_num) if use_cuda  else "cpu" #可以添加一个参数指定cuda
    logging.info(
        "Starting agent on device {}. Mode={}, logdir={}".format(
            device, flags.mode, flags.logdir
        )
    )
    
    polybeast_proc = mp.Process(
        target=polybeast.main,
        args=(flags, device),
        daemon=False,
    )
    pantheon_proc = mp.Process(
        target=pantheon_env.main, 
        args=(flags,), 
        daemon=False
    )
    
    polybeast_proc.start()
    #time.sleep(10)
    pantheon_proc.start()

    if mode=='train':
        # Training is driven by polybeast. Wait until it returns and then
        # kill pantheon_env.
        polybeast_proc.join()
        pantheon_proc.kill()
    else:
        # Testing is driven by pantheon_env. Wait for it to join and then
        # kill polybeast.
        pantheon_proc.join()
        polybeast_proc.kill()

    logging.info("Done {}".format(flags.mode))

    
def trace(flags):
    flags.mode = "trace"
    init_logdirs(flags)
    logging.info("Tracing model from checkpoint {}".format(flags.checkpoint))
    polybeast_proc = mp.Process(target=polybeast.main, args=(flags,), daemon=False)
    polybeast_proc.start()
    polybeast_proc.join()
    logging.info("Done tracing to {}".format(flags.traced_model))

def test_local(flags):
    init_logdirs(flags)

    if not os.path.exists(flags.traced_model):
        logging.info("Missing traced model, tracing first")
        trace(copy.deepcopy(flags))

    assert(flags.cc_env_mode == "local")

    logging.info("Starting local test, logdir={}".format(flags.logdir))
    pantheon_proc = mp.Process(target=pantheon_env.main, args=(flags,), daemon=False)
    pantheon_proc.start()
    pantheon_proc.join()
    logging.info("Done local test")

def main(flags):
    mode = flags.mode
    logging.info("Mode={}".format(mode))

    if mode == "train":
        # Train, trace, and then test
        run_remote(flags, mode='train')
        trace(flags)
        run_remote(flags, mode='test')
    elif mode == "test":
        # Only remote test
        run_remote(flags, mode='train')
    elif mode == 'test_local':
        test_local(flags)
    elif mode == "trace":
        trace(flags)
    else:
        raise RuntimeError("Unknown mode {}".format(mode))

    logging.info(
        "All done! Checkpoint: {}, traced model: {}".format(
            flags.checkpoint, flags.traced_model
        )
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = get_parser()
    flags = parser.parse_args()
    main(flags)
