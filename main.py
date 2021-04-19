"""Main script to run all components."""
import argparse
import jax.numpy as np
import numpy as onp
import time
from environment import Room, Roof, Rosenbrock
from fletcher_reeves import fletcher_reeves
from steepest_descent import steepest_descent
from neldor_mead import nelder_mead
from scipy.optimize import minimize
from gradient_descent import SGD
from gradient_descent_torch import torch_SGD
import torch

parser = argparse.ArgumentParser()

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
parser.add_argument('--num_bulbs', type=int, default=3)
parser.add_argument('--algorithm', type=str, default='steepest_descent')
parser.add_argument('--obj_weight', type=float, default=1.0)
parser.add_argument('--obj_function', type=str, default='simple_std')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--environment', type=str, default='rosen')
parser.add_argument('--vis', action='store_true')


def optimise_on_roof(args):
    onp.random.seed(42)
    num_bulbs = args.num_bulbs
    x0 = onp.random.randn(2 * num_bulbs)*2

    # "Intuition" based symmetric initialization
    if num_bulbs == 2:
        tmp = 1 + 0.1 * onp.random.randn(2)
        x0 = onp.hstack((tmp, -tmp))
        x0 = np.array(x0)

    # SciPy Nelder Mead
    # room = Roof(args.L, args.B, args.H, objective_type='simple_std')
    # res = minimize(room.J, x0, method='nelder-mead',
    #                options={'xatol': 1e-8, 'disp': True})
    # print(f"Minima at\n{room.to_pos(res.x).round(2)}")

    # SciPy Conjugate Gradient

    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
                obj_weight=args.obj_weight)
    room.show_plane(x0)
    print(f"Initialisation\n{room.to_pos(x0).round(2)}")
    print('Initial value: ', room.J(x0))
    res = minimize(room.J, x0, jac=room.gradient,
                   method='CG', options={'disp': True})
    print(f"Minima at\n{room.to_pos(res.x).round(2)}")
    room.show_plane(res.x)
    
    # Fletcher Reeves
    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
                obj_weight=args.obj_weight)
    x, k = fletcher_reeves(room.J,
                        room.gradient,
                        room.hessian,
                        x0, n_iter=1,
                        verbose=False,
                        )
    print(f"\nMinimum Value = {room.J(x)}")
    print(f"\nMinima at\n{room.to_pos(x).round(2)}")
    room.show_plane(x)


def one_call(args):
    onp.random.seed(args.random_seed)
    if args.environment == 'rosen':
        ix1 = -5 + onp.random.rand()*10
        ix2 = -5 + onp.random.rand()*10
        x0 = onp.array([ix1, ix2])
        room = Rosenbrock()
    # elif: args.environment == 'torch_env':
    #     num_bulbs = args.num_bulbs
    #     x0 = onp.random.randn(2 * num_bulbs)*2
    #     # "Intuition" based symmetric initialization
    #     if num_bulbs == 2:
    #         tmp = 1 + 0.1 * onp.random.randn(2)
    #         x0 = onp.hstack((tmp, -tmp))
    #         x0 = np.array(x0)

    #     room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
    #             obj_weight=args.obj_weight)    
    else:
        num_bulbs = args.num_bulbs
        x0 = onp.random.randn(2 * num_bulbs)*2
        # "Intuition" based symmetric initialization
        if num_bulbs == 2:
            tmp = 1 + 0.1 * onp.random.randn(2)
            x0 = onp.hstack((tmp, -tmp))
            x0 = np.array(x0)

        room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
                obj_weight=args.obj_weight)
        
    if args.vis:
        room.show_plane(x0)

    print(f"Initialisation\n{x0.round(2)}")
    print('Initial value: ', room.J(x0))
    start_time = time.time()
    if args.algorithm == 'scipy_cg':
        res = minimize(room.J, x0, jac=room.gradient,
                   method='CG', options={'disp': True})
    elif args.algorithm == 'scipy_bfgs':
        res = minimize(room.J, x0, method='BFGS', jac=room.gradient,
               options={'disp': True})
    elif args.algorithm == 'scipy_nelder_mead':
        res = minimize(room.J, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})
    elif args.algorithm == 'fletcher_reeves':
        x, count = fletcher_reeves(room.J,
                        room.gradient,
                        room.hessian,
                        x0, n_iter=1,
                        verbose=False,
                        )
    elif args.algorithm == 'nelder_mead':
        x, count = nelder_mead(room.J, x0)
    elif args.algorithm == 'gradient_descent':
        if args.environment != 'rosen':
            x, count = SGD(x0, room.objective_function, room.gradient)
        else:
            x, count = SGD(x0, room.objective_function, room.gradient, alpha=0.01) #TODO: [kushal] add parameters according to rosen
    elif args.algorithm == 'gradient_descent_torch':
        if args.environment != 'rosen':
            x0 = torch.FloatTensor(x0).view(-1,2)
            x0.requires_grad = True
            x, count = torch_SGD(x0, room, room.objective_function, iters=20000, lr=0.03)
        else:
            x0 = torch.FloatTensor(x0)
            x0.requires_grad = True
            x, count = torch_SGD(x0, room, room.objective_function, iters=20000, lr=1e-4, rosen_=1)
    else:
        raise NotImplementedError('Algorithms %s is not implemented.'
                                  % args.algorithm)
    
    runtime = time.time()
    runtime = runtime - start_time

    if args.algorithm.startswith('scipy'):
        final_x = res.x
        n_iter = res.nit
        final_obj = res.fun
    else:
        final_x = x
        n_iter = count
        final_obj = float(room.J(x))

    if args.vis:
        room.show_plane(final_x)

    print(f"Minima at\n{final_x.round(2)}")

    return runtime, n_iter, final_obj, final_x

def optimise_on_room(args):

    # Intialise X
    X_0 = np.zeros((args.num_bulbs, 3)).ravel()
    # Initalise Room
    # TODO: Replace hardcoded arguments with command line args
    room = Room(l=args.L, b=args.B, h=args.H, mesh_resolution=10,
                mesh_type='horizontal',
                plane_a=None, plane_b=None, plane_c=None, plane_d=None,
                plane_height=None, obj_weight=None, transform=False,
                objective_type='simple_min')

    if args.algorithm == 'steepest_descent':
        alpha_0 = np.ones_like(X_0) * 1e-2
        X_optimal = steepest_descent(X_0, alpha_0, room.objective_function,
                                     room.gradient)
    else:
        raise NotImplementedError('%s algorithm not implemented.' %
                                  args.algorithm)

 
if __name__ == '__main__':
    args = parser.parse_args()
    runtime, n_iter, final_obj, final_x = one_call(args)
    print('\nResults')
    print('Final Obj Value: %f\nRuntime: %fs\nNumber of iterations: %d' % (final_obj, runtime, n_iter))
    print('Minima at :', final_x)