import argparse
import os.path as path
from train import utils, constants
def add_pantheon_args(parser):
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
        "--sample",
        type=str,
        default='mean',
        choices=['mean', 'ave'],
        help="sample"
    )
    parser.add_argument(
        "--num_actors",
        type=int,
        default=0,
        help="Number of parallel actors for training. Default 0 starts one actor process per pantheon job.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=path.join(constants.SRC_DIR, "train/experiments.yml"),
        help="Configuration path for training",
    )
    parser.add_argument(
        "--test_config",
        type=str,
        default=path.join(constants.SRC_DIR, "train/experiments_test.yml"),
        help="Configuration path for testing",
    )
    parser.add_argument(
        "--run_times",
        type=int,
        default=1,
        help="Number of runs per job to average results over in test mode.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=path.join(constants.SRC_DIR, "train/logs/pantheon"),
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
        "--cc_env_time_window_ms",
        type=int,
        default=100,
        help="Window duration for time window aggregation",
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
    
def add_polybeast_args(parser):
    parser.add_argument("--xpid", default=None, help="Experiment id (default: None).")

    # Model output settings.
    parser.add_argument(
        "--checkpoint",  help="File to write checkpoints to."
    )

    # Model settings.
    parser.add_argument(
        "--observation_length",
        type=int,
        default=57,
        help="Length of the observation vector to be fed into the model.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size in FC model."
    )
    parser.add_argument(
        "--num_actions",
        type=int,
        default=11,
        help="Number of actions output by the policy.",
    )
    parser.add_argument(
        "--gpu_num",
        default=0,
        type=int,
        help="gpu number",
    )
    # Training settings.
    parser.add_argument(
        "--rpc_server_address",
        default="unix:/tmp/rl_server_path",
        type=str,
        help="Address to bind ActorPoolServer to. Could be either "
        "<host>:<port> or Unix domain socket path unix:<path>.",
    )
    parser.add_argument(
        "--savedir",
        default="~/palaas/torchbeast",
        help="Root dir where experiment data will be saved.",
    )
    parser.add_argument(
        "--total_steps",
        default=1000000,
        type=int,
        metavar="T",
        help="Total environment steps to train for",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, metavar="B", help="Learner batch size"
    )
    parser.add_argument(
        "--unroll_length",
        default=80,
        type=int,
        metavar="T",
        help="The unroll length (time dimension)",
    )
    parser.add_argument(
        "--num_learner_threads",
        default=2,
        type=int,
        metavar="N",
        help="Number of learner threads.",
    )
    parser.add_argument(
        "--num_inference_threads",
        default=2,
        type=int,
        metavar="N",
        help="Number of inference threads.",
    )
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument(
        "--use_lstm",
        type=utils.str2bool,
        default="true",
        help="Use LSTM in agent model.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Initial random seed for torch.random"
    )
    # Loss settings.
    parser.add_argument(
        "--entropy_cost", default=0.05, type=float, help="Entropy cost/multiplier."
    )
    parser.add_argument(
        "--baseline_cost", default=0.5, type=float, help="Baseline cost/multiplier."
    )
    parser.add_argument(
        "--discounting", default=0.99, type=float, help="Discounting factor."
    )
    parser.add_argument(
        "--reward_clipping",
        default="none",
        choices=["abs_one", "soft_asymmetric", "none"],
        help="Reward clipping.",
    )

    # Optimizer settings.
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="Learning rate.",
    )
    parser.add_argument(
        "--alpha", default=0.99, type=float, help="RMSProp smoothing constant."
    )
    parser.add_argument("--momentum", default=0, type=float, help="RMSProp momentum.")
    parser.add_argument("--epsilon", default=1e-5, type=float, help="RMSProp epsilon.")

    # Misc settings.
    parser.add_argument(
        "--write_profiler_trace",
        action="store_true",
        help="Collect and write a profiler trace " "for chrome://tracing/.",
    )
def add_common_args(parser):
    parser.add_argument("--base_logdir", type=str, default="logs")
    parser.add_argument(
        "--mode", default="train",
        choices=["train", "test", "trace", 'test_local'],
        help="test -> remote test, test_local -> local inference.",
    )
    parser.add_argument("--test_name", default="test", type=str)
    parser.add_argument(
        "--traced_model",
        help="File to write torchscript traced model to (for training) "
        "or read from (for local testing).",
    )
