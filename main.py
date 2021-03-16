"""Main script to run all components."""
import argparse
import jax.numpy as np
import numpy as onp

from environment import Room, Roof
from fletcher_reeves import fletcher_reeves
from scipy.optimize import minimize

parser = argparse.ArgumentParser()

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
...


def optimise(args):
    onp.random.seed(42)
    num_bulbs = 5
    x0 = onp.random.randn(2*num_bulbs)*2

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
    room = Roof(args.L, args.B, args.H, objective_type='simple_std')
    res = minimize(room.J, x0, jac=room.gradient,
                   method='CG', options={'disp': True})
    print(f"Minima at\n{room.to_pos(res.x).round(2)}")

    # Fletcher Reeves
    room = Roof(args.L, args.B, args.H, objective_type='simple_std')
    x = fletcher_reeves(room.J,
                        room.gradient,
                        room.hessian,
                        x0, n_iter=1,
                        verbose=True,
                        )
    print(f"\nMinimum Value = {room.J(x)}")
    print(f"\nMinima at\n{room.to_pos(x).round(2)}")


if __name__ == '__main__':
    args = parser.parse_args()
    optimise(args)
