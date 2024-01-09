# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import yaml
import string
import itertools
from datetime import datetime
import sysv_ipc
import itertools as it
import shlex
from train.constants import SRC_DIR, PANTHEON_ROOT
import subprocess

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# format 'format_string' but ignore keys that do not exist in 'key_dict'
def safe_format(format_string, key_dict):
    return string.Formatter().vformat(format_string, (), SafeDict(key_dict))

def rm_ipcms(actor_id, episode):
    try:
        shm_key_actor = (actor_id << 24) | episode
        shm_actor = sysv_ipc.SharedMemory(key=shm_key_actor)
        shm_actor.detach()
        shm_actor_clean_cmd = ["ipcrm", "-m", "{}".format(shm_actor.id)]
        p = subprocess.Popen(shm_actor_clean_cmd)
        p.wait()
    except sysv_ipc.ExistentialError:
        pass
    
    try:
        shm_key_link = (actor_id << 28) | episode
        shm_link = sysv_ipc.SharedMemory(key=shm_key_link)
        shm_link.detach()
        shm_link_clean_cmd = ["ipcrm", "-m", "{}".format(shm_link.id)]
        p = subprocess.Popen(shm_link_clean_cmd)
        p.wait()
    except sysv_ipc.ExistentialError:
        pass

def complete_cmd(flags, param_dict, cmd_tmpl, data_dir, actor_id, episode_id):
    if type(param_dict["bandwidth"]) == str:
        bdp = int(param_dict["bandwidth"].split('-')[0]) * 1000 * param_dict["delay"] * 2 / 8 / 1500
    else:
        bdp = param_dict["bandwidth"] * 1000 * param_dict["delay"] * 2 / 8 / 1500
    queueBDP = int(param_dict["queue"] * bdp)
    cmd_tmpl = safe_format(cmd_tmpl, param_dict)
    cmd = safe_format(cmd_tmpl, {"data_dir": data_dir, "queueBDP": queueBDP})
    extra_sender_args = " ".join(
        [
            "--cc_env_mode={}".format(flags.cc_env_mode),
            "--cc_env_rpc_address={}".format(flags.rpc_server_address),
            "--cc_env_model_file={}".format(flags.traced_model),
            "--cc_env_episode_id={}".format(episode_id),
            "--cc_env_actor_id={}".format(actor_id),
            "--cc_env_time_window_ms={}".format(flags.cc_env_time_window_ms),
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
            "-v={}".format(flags.loglevel),
            "--cc_env_bandwidth={}".format(param_dict['bandwidth']),
            "--cc_env_delay={}".format(param_dict['delay']),
            "--cc_env_loss_ratio={}".format(param_dict['loss_ratio']),
            "--cc_env_flows={}".format(param_dict['flows']),
        ]
    )
    #将字符串按照 shell 语法进行分割，生成一个 token 列表。  
    # #command_line = 'ls -l -a'; tokens = shlex.split(command_line);['ls', '-l', '-a']
    cmd =  shlex.split(cmd) + [
            "--run-times={}".format(flags.run_times), 
            "--actor_id={}".format(actor_id), 
            "--episode_id={}".format(episode_id), 
            '--extra-sender-args="{}"'.format(extra_sender_args)] + \
        (["--schemes=mvfst_rl"] if '--config_file' not in cmd else []) + \
        (["--do_log"] if flags.do_log == True else [])
    return cmd

def expand_matrix(matrix_cfg):
    input_list = []
    for variable, value_list in matrix_cfg.items():
        input_list.append([{variable: value} for value in value_list])

    ret = []
    for element in itertools.product(*input_list):
        tmp = {}
        for kv in element:
            tmp.update(kv)
        ret.append(tmp)

    return ret


def utc_date():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
