#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from os import path
import argparse
import logging
import subprocess
import random
import shlex
import shutil
import threading
import time
import numpy as np
import itertools as it
import json
#import train.merge_test_results_module as mtrm
import sysv_ipc
from train.constants import SRC_DIR, PANTHEON_ROOT
from train import common, utils

logging.basicConfig(level=logging.INFO)


def add_args(parser):
    #--merge test results args
    parser.add_argument(
            "--cc_scheme",
            type=str,
            default="mvfst_rl",
            help="Congestion control scheme to use",
    )
    parser.add_argument(
            "--num_columns",
            type=int,
            default=5,
            help="Number of columns in the merged test figure",
    )
    parser.add_argument(
            "--fig_col",
            type=float,
            default=24,
            help="Column size in the merged test figure",
    )
    parser.add_argument(
            "--fig_row",
            type= float, 
            default=24,
            help="Row size in the merged test figure",
    )
    parser.add_argument(
            "--dpi",
            type=int,
            default=100,
            help="big figure's dpi",
    )
    parser.add_argument(
            "--merge_results",
            action="store_true",
            help="do we merge the test results' figures into on big figure?",
    )
    
    #--
    parser.add_argument(
        "--do_log",
        action = 'store_true', 
        help="do we log during the training or test process",
    )
    #--
    parser.add_argument(
        "--num_actors",
        type=int,
        default=0,
        help="Number of parallel actors for training. Default 0 starts one actor process per pantheon job.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=path.join(SRC_DIR, "train/experiments.yml"),
        help="Configuration path for training",
    )
    parser.add_argument(
        "--test_config",
        type=str,
        default=path.join(SRC_DIR, "train/experiments_test.yml"),
        help="Configuration path for testing",
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        default=0,
        help="Maximum number of different Pantheon emulated experiments to use (0 for all)",
    )
    parser.add_argument(
        "--job_ids",
        type=str,
        default="",
        help="Comma separate list of job ids. If set, filter and train only on the specified pantheon jobs.",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="unix:/tmp/rl_server_path",
        help="RL server address - <host>:<port> or unix:<path>",
    )
    parser.add_argument(
        "--test_runs_per_job",
        type=int,
        default=3,
        help="Number of runs per job to average results over in test mode.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=path.join(SRC_DIR, "train/logs/pantheon"),
        help="Pantheon logs output directory",
    )
    parser.add_argument(
        "-v",
        "--loglevel",
        type=int,
        default=1,
        help="Verbose log-level for Pantheon sender",
    )

    # RLCongestionController args
    parser.add_argument(
        "--cc_env_mode",
        type=str,
        default="remote",
        choices=["local", "remote"],
        help="CongestionControlEnvConfig::Mode. Local only support testing.",
    )
    parser.add_argument(
        "--cc_env_agg",
        type=str,
        default="time",
        choices=["time", "fixed"],
        help="State aggregation type",
    )
    parser.add_argument(
        "--cc_env_time_window_ms",
        type=int,
        default=100,
        help="Window duration for time window aggregation",
    )
    parser.add_argument(
        "--cc_env_fixed_window_size",
        type=int,
        default=10,
        help="Window size for fixed window aggregation",
    )
    parser.add_argument(
        "--cc_env_use_state_summary",
        type=utils.str2bool,
        default="true",
        help="Whether to use state summary instead of raw states in observation (auto-enabled for time window aggregation)",
    )
    parser.add_argument(
        "--cc_env_history_size",
        type=int,
        default=8,
        help="Length of history (such as past actions) to include in observation",
    )
    parser.add_argument(
        "--cc_env_norm_ms",
        type=float,
        default=100.0,
        help="Norm factor for temporal fields",
    )
    parser.add_argument(
        "--cc_env_norm_bytes",
        type=float,
        default=1000.0,
        help="Norm factor for byte fields",
    )
    parser.add_argument(
        "--cc_env_actions",
        type=str,
        default="0,/2,/1.5,/1.2,/1.1,/1.05,*1.05,*1.1,*1.2,*1.5,*2",
        help="List of actions specifying how cwnd should be updated. First action should be 0 (no-op)",
    )
    parser.add_argument(
        "--cc_env_reward_throughput_factor",
        type=float,
        default=1.0,
        help="Throughput multiplier in reward",
    )
    parser.add_argument(
        "--cc_env_reward_delay_factor",
        type=float,
        default=0.01,
        help="Delay multiplier in reward",
    )
    parser.add_argument(
        "--cc_env_reward_packet_loss_factor",
        type=float,
        default=1.0,
        help="Packet loss multiplier in reward",
    )
    parser.add_argument(
        "--cc_env_reward_max_delay",
        type=utils.str2bool,
        default="true",
        help="Whether to take max delay over observations in reward (avg otherwise)",
    )


def train_run(flags, jobs, thread_id):
    """
    Each pantheon job runs for a default of 30s (max 60s allowed).
    We treat each such run as a separate episode for training and run
    randomly chosen job in parallel.
    """
    pantheon_env = get_pantheon_env(flags)
    episode = 1
    while True:
        # Pick a random experiment to run
        job_id = random.choice(range(len(jobs)))
        job_cfg, cmd_tmpl = jobs[job_id]
        # Expand data_dir in cmd template
        data_dir = path.join(
            flags.logdir, "train_tid{}_run{}_expt{}".format(thread_id, episode, job_id)
        )
        param_dict = job_cfg["params"].copy()
        for param in param_dict:
            value = param_dict[param]
            if type(value) is dict:
                param_dict[param] = random.sample(value.keys(), 1)[0]
            elif type(value) is list:
                param_dict[param] = random.sample(value, 1)[0]
        bdp = param_dict["bandwidth"] * 1000 * param_dict["delay"] * 2 / 8 / 1500
        param_dict["queue"] = int(param_dict["queue"] * bdp)
        cmd_tmpl = utils.safe_format(cmd_tmpl, param_dict)
        cmd = utils.safe_format(cmd_tmpl, {"data_dir": data_dir})
        cmd = update_cmd(cmd, flags, thread_id, episode, param_dict)

        logging.info(
            "Thread: {}, episode: {}, experiment: {}, cmd: {}".format(
                thread_id, episode, job_id, " ".join(cmd)
            )
        )
        p = subprocess.Popen(cmd, env=pantheon_env)
        p.wait()
        try:
            shm_key_actor = (thread_id << 24) | episode
            shm_actor = sysv_ipc.SharedMemory(key=shm_key_actor)
            shm_actor.detach()
            shm_actor_clean_cmd = ["ipcrm", "-m", "{}".format(shm_actor.id)]
            p = subprocess.Popen(shm_actor_clean_cmd)
            p.wait()
        except sysv_ipc.ExistentialError:
            pass
        
        try:
            shm_key_link = (thread_id << 28) | episode
            shm_link = sysv_ipc.SharedMemory(key=shm_key_link)
            shm_link.detach()
            shm_link_clean_cmd = ["ipcrm", "-m", "{}".format(shm_link.id)]
            p = subprocess.Popen(shm_link_clean_cmd)
            p.wait()
        except sysv_ipc.ExistentialError:
            pass
        
        episode += 1
        
        # Remove pantheon logs to free up space
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)


def test_run(flags, meta, jobs, thread_id):
    """
    Thread i runs jobs[i % len(jobs)] flags.test_runs_per_job times.
    """
    job_id = thread_id
    thread_id = thread_id % flags.num_actors + 1
    job_cfg, cmd_tmpl = jobs[job_id]
    episode = 1
    # Expand data_dir in cmd template
    data_dir = path.join(flags.logdir, "f{},b{:03d},q{:02d},l{},d{:02d}"
                         .format(job_cfg["params"]["flows"], 
                                 job_cfg["params"]["bandwidth"], 
                                 job_cfg["params"]["queue"], 
                                 job_cfg["params"]["loss_ratio"], 
                                 job_cfg["params"]["delay"]))
    bdp = job_cfg["params"]["bandwidth"] * 1000 * job_cfg["params"]["delay"] * 2 / 8 / 1500
    job_cfg["params"]["queue"] = int(job_cfg["params"]["queue"] * bdp)
    cmd_tmpl = utils.safe_format(cmd_tmpl, job_cfg["params"])
    cmd = utils.safe_format(cmd_tmpl, {"data_dir": data_dir})
    cmd = update_cmd(cmd, flags, thread_id, episode, job_cfg)
    # Run tests
    logging.info(
        "Test run: thread {} -> job {}, cmd: {}".format(
            thread_id, job_id, " ".join(cmd)
        )
    )
    pantheon_env = get_pantheon_env(flags)
    p = subprocess.Popen(cmd, env=pantheon_env)
    p.wait()

    # Run analysix
    analysis_cmd = [meta["analyze_path"], "--data-dir={}".format(data_dir)]
    logging.info(
        "Thread {}, job {}: Running analysis on {}, cmd: {}".format(
            thread_id, job_id, data_dir, " ".join(analysis_cmd)
        )
    )
    p = subprocess.Popen(analysis_cmd, env=pantheon_env)
    p.wait()
    
    try:
        shm_key_actor = (thread_id << 24) | episode
        shm_actor = sysv_ipc.SharedMemory(key=shm_key_actor)
        shm_actor.detach()
        shm_actor_clean_cmd = ["ipcrm", "-m", "{}".format(shm_actor.id)]
        p = subprocess.Popen(shm_actor_clean_cmd)
        p.wait()
    except sysv_ipc.ExistentialError:
        pass
    
    try:
        shm_key_link = (thread_id << 28) | episode
        shm_link = sysv_ipc.SharedMemory(key=shm_key_link)
        shm_link.detach()
        shm_link_clean_cmd = ["ipcrm", "-m", "{}".format(shm_link.id)]
        p = subprocess.Popen(shm_link_clean_cmd)
        p.wait()
    except sysv_ipc.ExistentialError:
        pass
    '''
    shutil.copyfile(
        path.join(data_dir, "pantheon_summary_mean.png"),
        path.join(flags.logdir, "test_expt{}.png".format(job_id)),
    )
    '''
    logging.info(
        "Test run finished for thread {}, job {}, Results in {}.".format(
            thread_id, job_id, data_dir
        )
    )


def run_pantheon_train(flags, jobs, num_threads, run_fn):
    logging.info(
        "Launching {} jobs over {} threads for {}.".format(
            len(jobs), num_threads, flags.mode
        )
    )

    threads = []
    for i in range(1, num_threads+1):
        thread = threading.Thread(target=run_fn, args=(flags, jobs, i))
        thread.start()
        threads.append(thread)
        # Stagger the beginning of each thread to avoid some errors due to
        # starting a bunch of Pantheon tunnels at once.
        time.sleep(1)

    for thread in threads:
        thread.join()
    logging.info("Done with {}.".format(flags.mode))


def run_pantheon_test(flags, meta, jobs, num_threads, run_fn):
    logging.info(
        "Launching {} jobs over {} threads for {}.".format(
            len(jobs), num_threads, flags.mode
        )
    )

    threads = []
    for job_id in range(len(jobs)):
        thread = threading.Thread(target=run_fn, args=(flags, meta, jobs, job_id))
        thread.start()
        threads.append(thread)
        # Stagger the beginning of each thread to avoid some errors due to
        # starting a bunch of Pantheon tunnels at once.
        time.sleep(1)

        if (job_id + 1) % num_threads == 0:
            for thread in threads:
                thread.join()
            threads = []
    logging.info("Done with {}.".format(flags.mode))
 #   if(flags.merge_results):
  #      logging.info("Merge results...")
  #      mtrm.merge_test_results(flags.cc_scheme, flags.logdir, flags.num_columns, flags.fig_col, flags.fig_row, flags.dpi)
    


def get_pantheon_emulated_jobs(flags):
    cfg = utils.parse_experiments(flags.train_config if flags.mode == "train" else flags.test_config)

    jobs = []
    for job_cfg in cfg["emu"]["jobs"]:
        cmd_tmpl = job_cfg["command"]
        # 1. Expand macros
        cmd_tmpl = utils.safe_format(cmd_tmpl, cfg["emu"]["macros"])
        # 2. Expand meta
        cmd_tmpl = utils.safe_format(cmd_tmpl, cfg["meta"])
        # 3. Expand parameter combinations for testing
        if flags.mode == "train":
            jobs.append((job_cfg, cmd_tmpl))
            with open(os.path.join(flags.logdir, 'train_env_params.json'), 'w') as f:
                json.dump(job_cfg['params'], f)
        else:
            param_keys = job_cfg["params"].keys()
            for param in param_keys:
                value = job_cfg["params"][param]
                if type(value) is dict:
                    job_cfg["params"][param] = list(value.keys())
            with open(os.path.join(flags.logdir, 'test_env_params.json'), 'w') as f:
                json.dump(job_cfg['params'], f)
            param_combs = it.product(*(job_cfg["params"][param] if type(job_cfg["params"][param]) is list
                                       else [job_cfg["params"][param]]
                                       for param in job_cfg["params"]))
            for param_values in param_combs:
                param_dict = dict(zip(param_keys, param_values))
                job_cfg_cpy = job_cfg.copy()
                job_cfg_cpy["params"] = param_dict
                jobs.append((job_cfg_cpy, cmd_tmpl))
    return cfg["meta"], jobs


def get_pantheon_env(flags):
    # $PATH override to put python2 first for Pantheon
    result = subprocess.run(
        ["dirname $(which python2)"], shell=True, stdout=subprocess.PIPE
    )
    python2_path = result.stdout.decode("utf-8").strip()
    logging.info("Located python2 in {}".format(python2_path))

    pantheon_env = os.environ.copy()
    pantheon_env["PATH"] = ":".join([python2_path, pantheon_env["PATH"]])
    return pantheon_env


def update_cmd(cmd, flags, actor_id, episode_id, params=None):
    # in train mode, params = cfg["emu"]["jobs"]["params"]
    if flags.mode == "train":
        schemes = "mvfst_rl"
        run_times = 1
    # in test mode, params = cfg["emu"]["jobs"]
    else:
        if flags.test_other:
            schemes = " ".join(params["schemes"])
        else:
            schemes = "mvfst_rl"
        params = params["params"]
        run_times = flags.test_runs_per_job
    extra_sender_args = None
    if not flags.test_other:
        extra_sender_args = " ".join(
            [
                "--cc_env_mode={}".format(flags.cc_env_mode),
                "--cc_env_rpc_address={}".format(flags.server_address),
                "--cc_env_model_file={}".format(flags.traced_model),
                "--cc_env_episode_id={}".format(episode_id),
                "--cc_env_actor_id={}".format(actor_id),
                "--cc_env_agg={}".format(flags.cc_env_agg),
                "--cc_env_time_window_ms={}".format(flags.cc_env_time_window_ms),
                "--cc_env_fixed_window_size={}".format(flags.cc_env_fixed_window_size),
                "--cc_env_use_state_summary={}".format(flags.cc_env_use_state_summary),
                "--cc_env_history_size={}".format(flags.cc_env_history_size),
                "--cc_env_norm_ms={}".format(flags.cc_env_norm_ms),
                "--cc_env_norm_bytes={}".format(flags.cc_env_norm_bytes),
                "--cc_env_actions={}".format(flags.cc_env_actions),
                "--cc_env_reward_throughput_factor={}".format(
                    flags.cc_env_reward_throughput_factor
                ),
                "--cc_env_reward_delay_factor={}".format(flags.cc_env_reward_delay_factor),
                "--cc_env_reward_packet_loss_factor={}".format(
                    flags.cc_env_reward_packet_loss_factor
                ),
                "--cc_env_reward_max_delay={}".format(flags.cc_env_reward_max_delay),
                "-v={}".format(flags.loglevel),
            ]
        )
        if params:
            extra_sender_args += " " + " ".join([
                "--cc_env_bandwidth={}".format(params['bandwidth']),
                "--cc_env_delay={}".format(params['delay']),
                "--cc_env_loss_ratio={}".format(params['loss_ratio']),
                "--cc_env_flows={}".format(params['flows']),
                ]) 
    #将字符串按照 shell 语法进行分割，生成一个 token 列表。  
    # #command_line = 'ls -l -a'; tokens = shlex.split(command_line);['ls', '-l', '-a']
    cmd =  shlex.split(cmd) + ["--run-times={}".format(run_times), 
            "--actor_id={}".format(actor_id), "--episode_id={}".format(episode_id)] + \
        (['--extra-sender-args="{}"'.format(extra_sender_args)] if extra_sender_args!=None else []) + \
        (["--schemes={}".format(schemes)] if '--flow-schemes' not in cmd else []) + \
        (["--do_log"] if flags.do_log == True else [])
    return cmd


def main(flags):
    meta, all_jobs = get_pantheon_emulated_jobs(flags)
    # Filter jobs to use for training
    if flags.job_ids and flags.mode == "train":
        job_ids = [int(job_id) for job_id in flags.job_ids.split(",")]
        jobs = [all_jobs[job_id] for job_id in job_ids]
        logging.info(
            "Filtered {} jobs corresponding to ids {}.".format(len(jobs), flags.job_ids)
        )
    else:
        jobs = all_jobs
        logging.info("Using all {} jobs.".format(len(jobs)))

    if flags.max_jobs > 0:
        logging.info("Filtering a maximum of {} jobs.".format(flags.max_jobs))
        jobs = jobs[: flags.max_jobs]

    if flags.mode == "train":
        # One thread / pantheon env per actor while training
        num_threads = flags.num_actors if flags.num_actors > 0 else len(jobs)
    else:
        # One thread per job to test
        num_threads = flags.num_actors if flags.num_actors < len(jobs) else len(jobs)

    if flags.mode == "train":
        run_pantheon_train(flags, jobs, num_threads, train_run)
    else:
        run_pantheon_test(flags, meta, jobs, num_threads, test_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pantheon Env Instances")
    common.add_args(parser)
    add_args(parser)
    flags = parser.parse_args()
    main(flags)
