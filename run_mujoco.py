#!/usr/bin/env python3

import sys

import tensorflow as tf
from mpi4py import MPI

import pposgd_simple
from utils import logger
from utils import mlp_policy
from utils.utils import set_global_seeds, make_mujoco_env, mujoco_arg_parser, get_cpu_per_task, pkl_res


def train(args):
    rank = MPI.COMM_WORLD.Get_rank()

    ncpu = get_cpu_per_task()
    ncpu //= 8

    sys.stdout.flush()

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    workerseed = int(args.seed)
    set_global_seeds(workerseed)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = make_mujoco_env(args.env, args.seed)
    running_scores = pposgd_simple.learn(env, policy_fn,
                                         timesteps_per_actorbatch=2048,
                                         optim_stepsize=3e-4, optim_batchsize=64,
                                         gamma=0.99, lam=0.95, schedule='linear',
                                         args=args
                                         )

    env.close()

    # Save result for run
    if MPI.COMM_WORLD.Get_rank() == 0:
        pkl_res(running_scores, args)


if __name__ == '__main__':
    args = mujoco_arg_parser().parse_args()
    print(args)

    train(args)
