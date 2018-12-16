import os
import pickle
import random

import gym
import numpy as np

from utils import logger
from utils.Monitor import Monitor


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))

    parser.add_argument('--sepg_lam',
                        help='Lambda. The weight between improving the next policy and keeping it close to the current policy',
                        type=float, default=1.2)
    parser.add_argument('--optim_epochs',
                        help='The maximum number of epochs to update the policy at each iteration',
                        type=int, default=30)
    parser.add_argument('--kl_threshold',
                        help='The KL threshold on the change in action distribution for each state (disaggregated constraint as referred to in SPU)',
                        type=float, default=0.05)
    parser.add_argument('--agg_kl_threshold',
                        help='The average KL threshold (aggregated constraint)',
                        type=float, default=0.05 / 1.3)

    return parser


def get_cpu_per_task():
    try:
        ncpu = os.environ['CPUS_PER_TASK']
        if ncpu == '':
            return 1
        else:
            return int(ncpu)
    except KeyError:
        return 1


def stringify_hp(args):
    return 'optim_epochs = ' + str(args.optim_epochs) + ', sepg_lam = ' + str(args.sepg_lam) \
           + ', kl_threshold = ' + str(args.kl_threshold) + ', agg_kl_threshold = ' + str(args.agg_kl_threshold)


def get_saved_path(args):
    hp_as_str = stringify_hp(args)

    dir = 'res/' + hp_as_str + '/' + args.env

    os.makedirs(dir, exist_ok=True)

    return dir + '/' + str(args.seed)


def get_graph_f(args):
    return get_saved_path(args) + '.png'


def get_pkl_f(args):
    return get_saved_path(args) + '.pkl'


def pkl_res(obj, args):
    # Format of path
    # is res -> hp string -> env -> {seed}.pkl

    pkl_f = get_pkl_f(args)

    with open(pkl_f, 'wb+') as f:
        pickle.dump(obj, f)


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)
