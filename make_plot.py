"""Final script to run all experiments"""
import argparse
import jax.numpy as np
import numpy as onp
import time
from main import one_call
import pickle as pkl
from plot import save_plot, save_combined
from scipy.spatial.distance import directed_hausdorff as conv_distance
from sympy.utilities.iterables import multiset_permutations
from environment import Roof

parser = argparse.ArgumentParser()

from experiment import get_save_string

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
parser.add_argument('--num_bulbs', type=int, default=3)
parser.add_argument('--algorithm', type=str, default='scipy_cg')
parser.add_argument('--obj_weight', type=float, default=1.0)
parser.add_argument('--obj_function', type=str, default='simple_std')
parser.add_argument('--seed_min', type=int, default=1)
parser.add_argument('--seed_max', type=int, default=11)
parser.add_argument('--environment', type=str, default='rosen')
parser.add_argument('--experiment_name', type=str, default='seed_range')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--save_dir', type=str, default='data/')
parser.add_argument('--plot_type', type=str, default='one')


def make_plot_one_seed_range(args):
    save_string = get_save_string(args)
    with open(save_string, 'rb') as f:
        results = pkl.load(f)
    metrics = ['final_obj', 'n_iter', 'runtime']

    x_range = list(range(args.seed_min, args.seed_max))

    for metric in metrics:
        save_plot_name = save_string.replace('.pkl', '_%s.png' % metric)

        save_plot(onp.around(results[metric], 3), x_range, 
        '%s for %s on %s problem.' % (metric, args.algorithm, args.environment),
            'Seeds', metric, save_plot_name)
        print('Metric %s:' % metric, results[metric])
    print('Metric %s:' % 'final_x', results['final_x'])

scipy_algo_name = {
    'fletcher_reeves' : 'scipy_cg',
    'gradient_descent': 'gradient_descent_torch',
    'nelder_mead' : 'scipy_nelder_mead'
    }

def make_plot_comparison(args):
    base_algo = args.algorithm
    scipy_algo = scipy_algo_name[base_algo]
    our_save_string = get_save_string(args)
    args.algorithm = scipy_algo
    scipy_save_string = get_save_string(args)
    
    metrics = ['final_obj', 'n_iter', 'runtime']
    x_range = list(range(args.seed_min, args.seed_max))

    with open(our_save_string, 'rb') as f:
        our_results = pkl.load(f)
    
    with open(scipy_save_string, 'rb') as f:
        scipy_results = pkl.load(f)
    
    for metric in metrics:
        save_plot_name = our_save_string.replace('.pkl', '_%s_comparison.png' % metric)        
        save_combined(val_ours=onp.around(our_results[metric], 3), val_scipy=onp.around(scipy_results[metric], 3),
                        x=x_range, title='%s comparison for %s on %s problem.' % (metric, args.algorithm, args.environment),
                        xtitle='Seeds', ytitle=metric, save_name=save_plot_name)
        print(metric, 'mean', 'ours', onp.mean(our_results[metric]), 'scipy', onp.mean(scipy_results[metric]))
        print(metric, 'std', 'ours', onp.std(our_results[metric]), 'scipy', onp.std(scipy_results[metric]))

def make_plot_distance(args):
    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
                obj_weight=args.obj_weight)
    base_algo = args.algorithm
    scipy_algo = scipy_algo_name[base_algo]
    our_save_string = get_save_string(args)
    args.algorithm = scipy_algo
    scipy_save_string = get_save_string(args)
    
    x_range = list(range(args.seed_min, args.seed_max))

    with open(our_save_string, 'rb') as f:
        our_results = pkl.load(f)
    
    with open(scipy_save_string, 'rb') as f:
        scipy_results = pkl.load(f)
    
    metric = 'final_x'
    val_ours = onp.around(our_results[metric], 3)
    val_scipy = onp.around(scipy_results[metric], 3)

    distance_vec = []
    for set1, set2 in zip(val_ours, val_scipy):
        set1 = onp.array(room.to_pos(set1))
        set2 = onp.array(room.to_pos(set2))
        set1 = set1.reshape(5, 2)
        set2 = set2.reshape(5, 2)
        # print(set1, set2)
        # dist = conv_distance(set1, set2)
        # print(dist)
        # distance_vec.append(dist[0])
        min_dist = np.inf
        indices = np.arange(5, dtype=int)
        for x in multiset_permutations(indices):
            set2perm = set2[x]
            print(set1, '\n', set2perm)
            dist = np.sqrt(np.linalg.norm(set1-set2perm))
            min_dist = min(min_dist, dist)
            print(dist, min_dist)
            
        distance_vec.append(min_dist)

    save_plot_name = our_save_string.replace('.pkl', '_%s_distance.png' % metric)        
    print(distance_vec)
    save_plot(distance_vec, x_range, 
            '%s for %s on %s problem.' % (metric, args.algorithm, args.environment),
                'Seeds', metric, save_plot_name)
    print('Metric %s:' % 'Distance', distance_vec)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.plot_type == 'one':
        make_plot_one_seed_range(args)
    elif args.plot_type == 'comparison':
        make_plot_comparison(args)
    elif args.plot_type == 'distance':
        make_plot_distance(args)
    else:
        raise NotImplementedError('%s plot type not implemented.' % args.plot_type)
