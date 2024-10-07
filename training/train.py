"""
Author: lee lizw_0304@163.com
Date: 2024-10-07 14:39:21
LastEditors: lee lizw_0304@163.com
LastEditTime: 2024-10-07 14:39:27
FilePath: /rangeplace_ros/training/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

# Warsaw University of Technology
# Train MinkLoc model

import argparse
import torch

from training.trainer import do_train
from misc.utils import TrainingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MinkLoc3Dv2 model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model-specific configuration file",
    )
    parser.add_argument("--resume", type=str, help="Path to the pretrain weights")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )  # 将 dest='debug' 移除

    args = parser.parse_args()
    print("Training config path: {}".format(args.config))
    print("Model config path: {}".format(args.model_config))
    print("Debug mode: {}".format(args.debug))

    params = TrainingParams(
        args.config, args.model_config, args.resume
    )  # 这里传递的 debug 是正确的
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    do_train(params)
