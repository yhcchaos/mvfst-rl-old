#!/bin/bash
mode=$1
total_steps=
base_logdir=
mkdir $base_logdir
checkpoint=
checkpoint_path=$base_logdir/checkpoint/$checkpoint.tar
echo base_logdir=$base_logdir
echo checkpoint=$checkpoint_path
echo total_steps=$total_steps
if [ $mode == "train" ]; then
	cp train/experiments.yml $base_logdir/experiments.yml
	GLOG_minloglevel=0 nohup python3 -m train.train --num_actions 11 --cc_env_history_size=20 --cc_env_reward_throughput_factor 1. --cc_env_reward_delay_factor 0.02 --cc_env_reward_packet_loss_factor 2.  --observation_length 48 --num_actors=10 --total_steps=$total_steps --base_logdir=$base_logdir  --mode=train >$base_logdir/ss_train.log 2>&1 &
else
	cp train/experiments_test.yml $base_logdir/experiments_test.yml
	GLOG_minloglevel=0 nohup python3 -m train.train --num_actions 11 --cc_env_history_size=20 --cc_env_reward_throughput_factor 1. --cc_env_reward_delay_factor 0.02 --cc_env_reward_packet_loss_factor 2.  --observation_length 48 --num_actors=10 --total_steps=5000000 --base_logdir=$base_logdir  --checkpoint=$checkpoint_path --mode=test --do_log >$base_logdir/ss_test.log 2>&1 &
fi
