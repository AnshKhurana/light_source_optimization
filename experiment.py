"""Final script to run all experiments"""
import argparse
import jax.numpy as np
import numpy as onp
import time
from main import one_call
import pickle as pkl
from plot import save_plot
import os
parser = argparse.ArgumentParser()

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
parser.add_argument('--num_bulbs', type=int, default=3)
parser.add_argument('--algorithm', type=str, default='scipy_cg')
parser.add_argument('--obj_weight', type=float, default=1.0)
parser.add_argument('--obj_function', type=str, default='simple_combined')
parser.add_argument('--seed_min', type=int, default=1)
parser.add_argument('--seed_max', type=int, default=11)
parser.add_argument('--environment', type=str, default='rosen')
parser.add_argument('--experiment_name', type=str, default='seed_range')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--save_dir', type=str, default='data/')

def check_and_make_dir(filepath):
    d_name = os.path.dirname(filepath)
    if not os.path.exists(d_name):
        os.makedirs(d_name)

def run_on_seed_range(args):
    # for this experiment we would be minimizing the variance
    args.obj_function = 'simple_std'
    # dict with list members
    results = dict()
    results['final_obj'] = []
    results['n_iter'] = []
    results['runtime'] = []
    results['final_x'] = []
    results['seed_min'] = args.seed_min
    results['seed_max'] = args.seed_max
    results['experiment_name'] = args.experiment_name
    
    save_string = get_save_string(args)
    check_and_make_dir(save_string)

    for seed in range(args.seed_min, args.seed_max):
        # change seed
        args.random_seed = seed
        runtime, n_iter, final_obj, final_x = one_call(args)
        results['runtime'].append(runtime)
        results['n_iter'].append(n_iter)
        results['final_x'].append(final_x)
        results['final_obj'].append(final_obj)
        # save intermediate results for checking values
        pkl.dump(results, open(save_string,'wb'))

def run_on_lambda_range(args):
    pass

def get_save_string(args):
    
    if args.environment == 'rosen':
        save_string = '%s_%s_%s_%d_%d.pkl' % (args.algorithm, args.environment,
                                 args.experiment_name,
                                 args.seed_min, args.seed_max)
    else:
        save_string = '%s_%s_%d_bulbs_%s_%d_%d.pkl' % (args.algorithm, 
        args.environment, args.num_bulbs, args.experiment_name, args.seed_min,
                        args.seed_max)
    save_string = os.path.join(args.save_dir, save_string)

    return save_string

if __name__=='__main__':
    args = parser.parse_args()
    if args.experiment_name=='seed_range':
        run_on_seed_range(args)
    else:
        raise NotImplementedError('Experiment %s is not implemented.' 
        % args.experiment_name)