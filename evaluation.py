"""For all metrics, comparison and plotting code."""

import argparse
import jax.numpy as np
import numpy as onp

from environment import Room, Roof
from fletcher_reeves import fletcher_reeves
from scipy.optimize import minimize
import time
parser = argparse.ArgumentParser()

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
parser.add_argument('--num_bulbs', type=int, default=5)
parser.add_argument('--obj_function', type=str, default='simple_combined')


SciPy_nm, SciPy_cg, FR = [], [], []
SciPy_cg_nit, SciPy_nm_nit, FR_nit = [], [], []
SciPy_cg_time, SciPy_nm_time, FR_time = [],[],[]

def optimise(args, i):
    global SciPy_cg, SciPy_nm, FR, SciPy_cg_nit, SciPy_nm_nit,\
     FR_nit, SciPy_cg_time, SciPy_nm_time, FR_time
    num_bulbs = args.num_bulbs
    onp.random.seed(42)

    x0 = onp.random.randn(2*num_bulbs)*2
    # "Intuition" based symmetric initialization
    if num_bulbs == 2:
        tmp = 1 + 0.1 * onp.random.randn(2)
        x0 = onp.hstack((tmp, -tmp))
        x0 = np.array(x0)


    # SciPy Conjugate Gradient
    start = time.time()
    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
                obj_weight=i)
    res = minimize(room.J, x0, jac=room.gradient,
                   method='CG', options={'disp': True})
    SciPy_cg.append(res.fun)
    SciPy_cg_nit.append(res.nit)
    print(f"Minima at\n{room.to_pos(res.x).round(2)}")
    end = time.time()
    SciPy_cg_time.append(end-start)
    # Fletcher Reeves
    start = time.time()
    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function,
                obj_weight=i)
    x, count = fletcher_reeves(room.J,
                        room.gradient,
                        room.hessian,
                        x0, n_iter=1,
                        verbose=False,
                        )
    FR.append(float(room.J(x)))
    FR_nit.append(count)
    print(f"\nMinimum Value = {room.J(x)}")
    print(f"\nMinima at\n{room.to_pos(x).round(2)}")
    end = time.time()
    FR_time.append(end-start)



if __name__ == '__main__':
    args = parser.parse_args()
    for i in [0, 0.5]:
        optimise(args,i)
        print('Loss:', SciPy_cg, FR)
        print('NIT:', SciPy_cg_nit, FR_nit)
        print('Time:',SciPy_cg_time, FR_time)
        import pickle as pkl
        pkl.dump([SciPy_cg, FR, SciPy_cg_nit, FR_nit, SciPy_cg_time, FR_time]
            , open('5bulbs_gamma.pkl','wb'))

