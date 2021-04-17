"""Final script to run all experiments"""
import argparse
import jax.numpy as np
import numpy as onp
import time
from main import one_call
import pickle as pkl
from plot import save_plot
parser = argparse.ArgumentParser()
from experiment import get_save_string

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
if __name__ == '__main__':
    args = parser.parse_args()
    make_plot_one_seed_range(args)