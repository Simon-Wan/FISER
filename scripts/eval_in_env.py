import argparse
import os
from evaluators.rp_eval import rp_eval
from evaluators.e2e_eval import e2e_eval

import traceback
import ipdb
import sys


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    ipdb.pm()


sys.excepthook = info

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment config
    parser.add_argument('--experiment_name', default='v2', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--pipeline', choices=['r', 'p', 'rp', 'e2e'], type=str)
    parser.add_argument('--mid', choices=['none', 'qsvo', 'obj', 'qsvo+obj'])

    # path directory
    parser.add_argument('--data_path', default='../HandMeThat/datasets/v2', type=str)
    parser.add_argument('--data_dir_name', default='HandMeThat_with_expert_demonstration', type=str)
    parser.add_argument('--data_info_name', default='HandMeThat_data_info.json', type=str)
    parser.add_argument('--save_path', default='checkpoints', type=str)

    # data config
    parser.add_argument('--goal', default=-1, type=int)
    parser.add_argument('--level', default='level0', type=str)
    parser.add_argument('--quest_type', default='bring_me', type=str)
    parser.add_argument('--num_files', default=100, type=int)

    # model config
    parser.add_argument('--d_model', default=32, type=int)
    parser.add_argument('--N_qsvo', default=3, type=int)
    parser.add_argument('--N_obj', default=3, type=int)
    parser.add_argument('--N_action', default=3, type=int)

    # evaluator config
    parser.add_argument('--r_epoch_id', default=0, type=int)
    parser.add_argument('--p_epoch_id', default=0, type=int)
    parser.add_argument('--e2e_epoch_id', default=0, type=int)
    parser.add_argument('--e2e_seed', default=0, type=int)
    parser.add_argument('--r_seed', default=0, type=int)
    parser.add_argument('--p_seed', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--v1', action="store_true")
    parser.add_argument('--gen', action="store_true")
    
    parser.add_argument('--gt_qsvo', action="store_true")
    parser.add_argument('--gt_obj', action="store_true")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # GOALS = [2, 3, 24, 25, 26, 30, 31, 32, 33, 35, 36, 46, 52, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66]

    if args.v1 == True:
        args.data_path = '../HandMeThat/datasets/v1'
    if args.pipeline == 'rp':
        rp_eval(args)
    elif args.pipeline == 'r':
        pass
    elif args.pipeline == 'p':
        pass
    elif args.pipeline == 'e2e':
        e2e_eval(args)
    else:
        raise ValueError('Invalid pipeline argument!')


