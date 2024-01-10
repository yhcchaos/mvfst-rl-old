#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from os import path
from train import arg_parser
import argparse
import logging
import subprocess
import random
import yaml
import shutil
import threading
import logging
import numpy as np
from fractions import Fraction
import time
import itertools as it
#import train.merge_test_results_module as mtrm
from train.constants import SRC_DIR, PANTHEON_ROOT
from train import utils

logging.basicConfig(level=logging.INFO)

def train_run(flags, jobs, probs, actor_id):
    # prob is only used when flags.sample=='ave' 
    episode = 1
    while True:
        if flags.sample=='ave':
            if len(probs) == 0:
                job_id = np.random.choice(range(len(jobs)))
            else:
                job_id = np.random.choice(range(len(jobs)), p=probs)
            job_cfg, cmd_tmpl = jobs[job_id]
            param_dict = job_cfg["params"].copy()
            for param in param_dict:
                value = param_dict[param]
                assert(type(value)==list)
                param_dict[param] = random.sample(value, 1)[0]
        else:
            job_id = np.random.choice(range(len(jobs)))
            param_dict, cmd_tmpl = jobs[job_id]
        # Expand data_dir in cmd template
        data_dir = path.join(
            flags.logdir, "train_tid{}_run{}_expt{}".format(actor_id, episode, job_id)
        )
        
        cmd = utils.complete_cmd(flags, param_dict, cmd_tmpl, data_dir, actor_id, episode)
        logging.info(
            "actor_id: {}, episode: {}, job_id: {}, cmd: {}".format(
                actor_id, episode, job_id, " ".join(cmd)
            )
        )
        p = subprocess.Popen(cmd)
        p.wait()
        utils.rm_ipcms(actor_id, episode)
        episode += 1
        # Remove pantheon logs to free up space
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)

def test_run(flags, meta, jobs, job_id):
    actor_id = job_id % flags.num_actors + 1
    param_dict, cmd_tmpl = jobs[job_id]
    episode = 1
    # Expand data_dir in cmd template
    data_dir = path.join(flags.logdir, "f{},b{},q{:02d},l{:0<6.4f},d{:02d}".format(
                                        param_dict["flows"], 
                                        param_dict["bandwidth"], 
                                        param_dict["queue"], 
                                        param_dict["loss_ratio"], 
                                        param_dict["delay"]))
    cmd = utils.complete_cmd(flags, param_dict, cmd_tmpl, data_dir, actor_id, episode)
    # Run tests
    logging.info(
        "Test run: thread {} -> job {}, cmd: {}".format(
            actor_id, job_id, " ".join(cmd)
        )
    )
    p = subprocess.Popen(cmd)
    p.wait()

    # Run analysix
    analysis_cmd = [meta["analyze_path"], "--data-dir={}".format(data_dir)]
    logging.info(
        "Thread {}, job {}: Running analysis on {}, cmd: {}".format(
            actor_id, job_id, data_dir, " ".join(analysis_cmd)
        )
    )
    p = subprocess.Popen(analysis_cmd)
    p.wait()
    utils.rm_ipcms(actor_id, episode)
    episode += 1
    logging.info(
        "Test run finished for thread {}, job {}, Results in {}.".format(
            actor_id, job_id, data_dir
        )
    )

def main(flags):
    env_cfg_path = flags.train_config if flags.mode == "train" else flags.test_config
    with open(env_cfg_path) as cfg:
        env_cfg = yaml.safe_load(
            utils.safe_format(cfg.read(), {"src_dir": SRC_DIR, "pantheon_root": PANTHEON_ROOT})
        )
    jobs = []
    probs = []
    for job_cfg in env_cfg["emu"]["jobs"]:
        cmd_tmpl = job_cfg["command"]
        # 1. Expand macros
        cmd_tmpl = utils.safe_format(cmd_tmpl, env_cfg["emu"]["macros"])
        # 2. Expand meta
        cmd_tmpl = utils.safe_format(cmd_tmpl, env_cfg["meta"])
        if flags.mode == "train" and flags.sample=='ave':
            if 'prob' in job_cfg:
                probs.append(float(Fraction(job_cfg['prob'])))
            jobs.append((job_cfg, cmd_tmpl))
        else:
            # 3. Expand parameter combinations for testing
            param_keys = job_cfg["params"].keys()
            param_combs = it.product(*(job_cfg["params"][param] for param in job_cfg["params"]))
            for param_values in param_combs:
                param_dict = dict(zip(param_keys, param_values))
                jobs.append((param_dict, cmd_tmpl))

    if flags.mode == "test" and flags.num_actors > len(jobs):
        flags.num_actors = len(jobs)
    logging.info(
        "Launching {} jobs over {} threads for {}.".format(
            len(jobs), flags.num_actors, flags.mode
        )
    )
    threads = []
    if flags.mode == "train":
        for actor_id in range(1, flags.num_actors +1):
            thread = threading.Thread(target=train_run, args=(flags, jobs, probs, actor_id))
            thread.start()
            threads.append(thread)
            # Stagger the beginning of each thread to avoid some errors due to
            # starting a bunch of Pantheon tunnels at once.
            time.sleep(1)
        for thread in threads:
            thread.join()
    else:
        for job_id in range(len(jobs)):
            thread = threading.Thread(target=test_run, args=(flags, env_cfg['meta'], jobs, job_id))
            thread.start()
            threads.append(thread)
            # Stagger the beginning of each thread to avoid some errors due to
            # starting a bunch of Pantheon tunnels at once.
            time.sleep(1)
            if (job_id + 1) % flags.num_actors == 0:
                for thread in threads:
                    thread.join()
                threads = []
        logging.info("Done with {}.".format(flags.mode))
        #if(flags.merge_results):
        #    logging.info("Merge results...")
        #    mtrm.merge_test_results(flags.cc_scheme, flags.logdir, flags.num_columns, flags.fig_col, flags.fig_row, flags.dpi)
          
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pantheon Env Instances")
    arg_parser.add_pantheon_args(parser)
    flags = parser.parse_args()
    main(flags)