#!/bin/bash
mode=$1
if [ $mode == "train" ]; then
	GLOG_minloglevel=0 nohup python3 -m train.train --num_actions 11 --cc_env_history_size=20 --cc_env_reward_throughput_factor 1. --cc_env_reward_delay_factor 0.015 --cc_env_reward_packet_loss_factor 2.  --observation_length 48 --num_actors=10 --total_steps=20000000 --base_logdir=logs/e9m12_1216_1-0.02*DeltaDivSumDelta*f-2_steps2000wan_f-exp_b12-192-exp_q1bdp_l0.01_d5-85-5_rt30_a10  --checkpoint=logs/e9m12_1216_1-0.02*DeltaDivSumDelta*f-2_steps2000wan_f-exp_b12-192-exp_q1bdp_l0.01_d5-85-5_rt30_a10/checkpoints/checkpoint.tar9600640.tar --mode=train >ss_e9m12.log 2>&1 &
else
	GLOG_minloglevel=0 nohup python3 -m train.train --num_actions 11 --cc_env_history_size=20 --cc_env_reward_throughput_factor 1. --cc_env_reward_delay_factor 0.02 --cc_env_reward_packet_loss_factor 2.  --observation_length 48 --num_actors=10 --total_steps=5000000 --base_logdir=logs/e9m12_1216_1-0.02*DeltaDivSumDelta*f-2_steps2000wan_f-exp_b12-192-exp_q1bdp_l0.01_d5-85-5_rt30_a10  --checkpoint=logs/e9m12_1216_1-0.02*DeltaDivSumDelta*f-2_steps2000wan_f-exp_b12-192-exp_q1bdp_l0.01_d5-85-5_rt30_a10/checkpoints/checkpoint.tar9600640.tar --mode=test --do_log --merge_results >ss_test_e9m6.log 2>&1 &
fi

