"""Main script to run all components."""
import argparse
import jax.numpy as np
import numpy as onp

from environment import Room, Roof
from fletcher_reeves import fletcher_reeves
from steepest_descent import steepest_descent
# from nelder_mead import neldor_mead
from scipy.optimize import minimize

parser = argparse.ArgumentParser()

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
parser.add_argument('--num_bulbs', type=int, default=3)
parser.add_argument('--algorithm', type=str, default='steepest_descent')
parser.add_argument('--obj_weight', type=float, default=1.0)
parser.add_argument('--obj_function', type=str, default='simple_combined')




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
    optimise_on_roof(args)
