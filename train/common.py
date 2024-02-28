# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse


def add_args(parser):
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test", "test_local", "test_remote", "trace"],
        help="test -> remote test, test_local -> local inference., test_remote->test in real world",
    )
    parser.add_argument(
        "--traced_model",
        default="traced_model.pt",
        help="File to write torchscript traced model to (for training) "
        "or read from (for local testing).",
    )
    parser.add_argument(
        "--test_other",
        action="store_true",
        help="test other cc schemes"
    )