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

from gala.gpu_gossip_buffer import GossipBuffer
from gala.graph_manager import FullyConnectedGraph as Graph

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
    parser.add_argument("--base_tmpdir", type=str, default="/dev/shm/pantheon-data-" + getpass.getuser())

    return parser


def init_logdirs(flags):
    pantheon_tmpdir = os.path.join('/dev/shm/', 'pantheon-tmp-' + getpass.getuser())
    if os.path.exists(pantheon_tmpdir):
        shutil.rmtree(pantheon_tmpdir)
    if os.path.exists(flags.base_tmpdir):
        shutil.rmtree(flags.base_tmpdir)
    #flags.logdir = os.path.join(flags.base_tmpdir, flags.mode)
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
    checkpoint_dir = os.path.join(flags.base_logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if( flags.checkpoint is None ):
        flags.checkpoint = os.path.join(checkpoint_dir, "checkpoint.tar")
    #flags.traced_model = os.path.join(flags.base_logdir, "traced_model.pt")

    if flags.mode != "train":
        assert os.path.exists(
            flags.checkpoint
        ), "Checkpoint {} missing in {} mode".format(flags.checkpoint, flags.mode)

def move_logs():
    copy_tree(flags.base_tmpdir, flags.base_logdir)
    shutil.rmtree(flags.base_tmpdir)

def make_gossip_buffer(flags, num_agents, mng, device):
    """
    Shared gossip buffer for GALA mode.
    """
    if num_agents <= 1:
        return None, None

    # Make topology
    topology = []
    for rank in range(num_agents):
        graph = Graph(rank, num_agents, peers_per_itr=flags.num_gala_peers)
        topology.append(graph)

    # Initialize parameter buffer
    flags.observation_shape = (1, 1, flags.observation_length)
    model = polybeast.Net(
        observation_shape=flags.observation_shape,
        hidden_size=flags.hidden_size,
        num_actions=flags.num_actions,
        use_lstm=flags.use_lstm,
    )
    model.to(device)

    # Keep track of local iterations since learner's last sync
    sync_list = mng.list([0 for _ in range(num_agents)])
    # Used to ensure proc-safe access to agents' message-buffers
    buffer_locks = mng.list([mng.Lock() for _ in range(num_agents)])
    # Used to signal between processes that message was read
    read_events = mng.list(
        [mng.list([mng.Event() for _ in range(num_agents)]) for _ in range(num_agents)]
    )
    # Used to signal between processes that message was written
    write_events = mng.list(
        [mng.list([mng.Event() for _ in range(num_agents)]) for _ in range(num_agents)]
    )

    # Need to maintain a reference to all objects in main processes
    _references = [topology, model, buffer_locks, read_events, write_events, sync_list]
    gossip_buffer = GossipBuffer(
        topology,
        model,
        buffer_locks,
        read_events,
        write_events,
        sync_list,
        sync_freq=flags.sync_freq,
    )
    return gossip_buffer, _references


def run_remote(flags, train=True):
    flags.mode = "train" if train else "test"
    # flags.disable_cuda = not train
    flags.cc_env_mode = "remote"

    proc_manager = mp.Manager()
    barrier = None
    shared_gossip_buffer = None
    cuda = not flags.disable_cuda and torch.cuda.is_available()

    num_agents = 1
    if train and flags.num_gala_agents > 1:
        # In GALA mode. Start multiple replicas of the polybeast-pantheon setup.
        num_agents = flags.num_gala_agents
        logging.info("In GALA mode, will start {} agents".format(num_agents))
        barrier = proc_manager.Barrier(num_agents)

        # Shared-gossip-buffer on GPU-0
        device = torch.device("cuda:0" if cuda else "cpu")
        shared_gossip_buffer, _references = make_gossip_buffer(
            flags, num_agents, proc_manager, device
        )

    base_logdir = flags.base_logdir
    polybeast_proc = []
    pantheon_proc = []
    for rank in range(num_agents):
        flags.base_logdir = (
            os.path.join(base_logdir, "gala_{}".format(rank))
            if num_agents > 1
            else base_logdir
        )
        init_logdirs(flags)

        # Unix domain socket path for RL server address, one per GALA agent.
        address = "/tmp/rl_server_path_{}".format(rank)
        try:
            os.remove(address)
        except OSError:
            pass
        flags.address = "unix:{}".format(address)
        flags.server_address = flags.address

        # Round-robin device assignment
        device = "cuda:{}".format(rank % torch.cuda.device_count()) if cuda else "cpu"

        logging.info(
            "Starting agent {} on device {}. Mode={}, logdir={}".format(
                rank, device, flags.mode, flags.logdir
            )
        )
        polybeast_proc.append(
            mp.Process(
                target=polybeast.main,
                args=(flags, rank, barrier, device, shared_gossip_buffer),
                daemon=False,
            )
        )
        pantheon_proc.append(
            mp.Process(target=pantheon_env.main, args=(flags,), daemon=False)
        )
        polybeast_proc[rank].start()
        pantheon_proc[rank].start()

    if train:
        # Training is driven by polybeast. Wait until it returns and then
        # kill pantheon_env.
        for rank in range(num_agents):
            polybeast_proc[rank].join()
            pantheon_proc[rank].kill()
    else:
        # Testing is driven by pantheon_env. Wait for it to join and then
        # kill polybeast.
        for rank in range(num_agents):
            pantheon_proc[rank].join()
            polybeast_proc[rank].kill()

    move_logs()
    logging.info("Done {}".format(flags.mode))


def test_local(flags):
    flags.mode = "test"
    init_logdirs(flags)

    if not os.path.exists(flags.traced_model):
        logging.info("Missing traced model, tracing first")
        trace(copy.deepcopy(flags))

    flags.cc_env_mode = "local"

    logging.info("Starting local test, logdir={}".format(flags.logdir))
    pantheon_proc = mp.Process(target=pantheon_env.main, args=(flags,), daemon=False)
    pantheon_proc.start()
    pantheon_proc.join()
    move_logs()
    logging.info("Done local test")


def trace(flags):
    flags.mode = "trace"
    init_logdirs(flags)

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
    '''
Namespace(address='unix:/tmp/rl_server_path', alpha=0.99, base_logdir='experiments/1-0.02-2_steps50000_f1b12q60lr0.06d5-100-5_obs14', base_tmpdir='/dev/shm/pantheon-data-yhchaos', baseline_cost=0.5, batch_size=1, cc_env_actions='0,/2,/1.5,/1.2,/1.1,/1.05,*1.05,*1.1,*1.2,*1.5,*2', cc_env_agg='time', cc_env_fixed_window_size=10, cc_env_history_size=20, cc_env_mode='remote', cc_env_norm_bytes=1000.0, cc_env_norm_ms=100.0, cc_env_reward_delay_factor=0.02, cc_env_reward_max_delay=True, cc_env_reward_packet_loss_factor=2.0, cc_env_reward_throughput_factor=1.0, cc_env_time_window_ms=100, cc_env_use_state_summary=True, checkpoint='checkpoint.tar', disable_cuda=False, discounting=0.99, entropy_cost=0.05, epsilon=1e-05, hidden_size=512, job_ids='', learning_rate=0.001, logdir='/home/yhchaos/mvfst-rl/train/logs/pantheon', loglevel=1, max_jobs=0, mode='train', momentum=0, num_actions=11, num_actors=20, num_gala_agents=1, num_gala_peers=1, num_inference_threads=2, num_learner_threads=2, observation_length=14, reward_clipping='none', savedir='~/palaas/torchbeast', seed=1234, server_address='unix:/tmp/rl_server_path', sync_freq=0, test_config='/home/yhchaos/mvfst-rl/train/experiments_test.yml', test_runs_per_job=3, total_steps=50000, traced_model='traced_model.pt', train_config='/home/yhchaos/mvfst-rl/train/experiments.yml', unroll_length=80, use_lstm=True, write_profiler_trace=False, xpid=None)
'''
    main(flags)
