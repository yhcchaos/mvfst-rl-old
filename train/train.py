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
import os
import getpass
import shutil
from distutils.dir_util import copy_tree
import sys

from train import polybeast, pantheon_env, common, utils
from train.constants import THIRD_PARTY_ROOT

sys.path.append(THIRD_PARTY_ROOT)
logging.basicConfig(level=logging.INFO)
os.environ["OMP_NUM_THREADS"] = "1"

def get_parser():
    parser = argparse.ArgumentParser()
    common.add_args(parser)

    polybeast_parser = parser.add_argument_group("polybeast")
    polybeast.add_args(polybeast_parser)

    pantheon_parser = parser.add_argument_group("pantheon_env")
    pantheon_env.add_args(pantheon_parser)

    parser.add_argument("--base_logdir", type=str, default="logs")
    return parser


def init_logdirs(flags, other= False):
    pantheon_tmpdir = os.path.join('/dev/shm/', 'pantheon-tmp-' + getpass.getuser())
    if os.path.exists(pantheon_tmpdir):
        shutil.rmtree(pantheon_tmpdir)
    flags.savedir = os.path.join(os.path.join(flags.base_logdir, flags.mode), "torchbeast")
    flags.logdir=flags.savedir
    # Clean run for test mode
    if flags.mode != "train" and os.path.exists(flags.logdir):
        shutil.rmtree(flags.logdir)

    if flags.mode == "trace":
        os.makedirs(flags.base_logdir, exist_ok=True)
    else:
        os.makedirs(flags.logdir, exist_ok=True)
        os.makedirs(flags.savedir, exist_ok=True)

    if(flags.mode == "train"):
        checkpoint_dir = os.path.join(flags.base_logdir, "checkpoints")
        if os.path.exists(os.path.join(checkpoint_dir, "checkpoints")):
            assert len(os.listdir(checkpoint_dir)) == 0, "checkpoints directory not empty"
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if(flags.checkpoint == None):
            flags.checkpoint = os.path.join(checkpoint_dir, "checkpoint.tar")
    logging.error(flags.mode)
    if not flags.test_other:
        flags.traced_model = os.path.join(flags.base_logdir, "traced_model.pt")
        if flags.mode != "train":
            assert os.path.exists(
                flags.checkpoint
            ), "Checkpoint {} missing in {} mode".format(flags.checkpoint, flags.mode)



def run_remote(flags, train=True):
    flags.mode = "train" if train else "test"
    # flags.disable_cuda = not train
    flags.cc_env_mode = "remote"
    cuda = not flags.disable_cuda and torch.cuda.is_available()
    init_logdirs(flags)
    # Unix domain socket path for RL server address, one per GALA agent.
    address = "/tmp/rl_server_path"
    try:
        os.remove(address)
    except OSError:
        pass
    flags.address = "unix:{}".format(address)
    flags.server_address = flags.address
        # Round-robin device assignment
    device = "cuda:0" if cuda else "cpu" #可以添加一个参数指定cuda
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
    pantheon_proc.start()

    if train:
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


def test_local(flags):
    flags.mode = "test"
    init_logdirs(flags)

    if not flags.test_other and not os.path.exists(flags.traced_model):
        logging.info("Missing traced model, tracing first")
        trace(copy.deepcopy(flags))

    flags.cc_env_mode = "local"

    logging.info("Starting local test, logdir={}".format(flags.logdir))
    pantheon_proc = mp.Process(target=pantheon_env.main, args=(flags,), daemon=False)
    pantheon_proc.start()
    pantheon_proc.join()
    logging.info("Done local test")
    
def trace(flags):
    flags.mode = "trace"
    init_logdirs(flags, True)

    logging.info("Tracing model from checkpoint {}".format(flags.checkpoint))
    polybeast_proc = mp.Process(target=polybeast.main, args=(flags,), daemon=False)
    polybeast_proc.start()
    polybeast_proc.join()
    logging.info("Done tracing to {}".format(flags.traced_model))


def main(flags):
    mode = flags.mode
    logging.info("Mode={}".format(mode))

    if mode == "train":
        # Train, trace, and then test
        run_remote(flags, train=True)
        trace(flags)
        run_remote(flags, train=False)
    elif mode == "test":
        # Only remote test
        run_remote(flags, train=False)
    elif mode == "test_local":
        # Only local test
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
