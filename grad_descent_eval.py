"""For all metrics, comparison and plotting code."""

import argparse
import numpy as np
import torch
from environment import Room, Roof
from fletcher_reeves import fletcher_reeves
from scipy.optimize import minimize
from neldor_mead import nelder_mead
from gradient_descent import SGD
from gradient_descent_torch import torch_SGD
import time
parser = argparse.ArgumentParser()

parser.add_argument('-L', type=float, default=10)
parser.add_argument('-B', type=float, default=20)
parser.add_argument('-H', type=float, default=15)
parser.add_argument('--num_bulbs', type=int, default=4)
parser.add_argument('--obj_function', type=str, default='simple_std')


SciPy_nm, SciPy_cg, FR = [], [], []
SciPy_cg_nit, SciPy_nm_nit, FR_nit = [], [], []
SciPy_cg_time, SciPy_nm_time, FR_time = [],[],[]

def optimise(args,s):
    global SciPy_cg, SciPy_nm, FR, SciPy_cg_nit, SciPy_nm_nit,\
     FR_nit, SciPy_cg_time, SciPy_nm_time, FR_time
    num_bulbs = args.num_bulbs
    np.random.seed(s)

    x0 = np.random.randn(2*num_bulbs)*2


    # SciPy Conjugate Gradient
    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function)
    
    xj = torch.FloatTensor(x0).view(num_bulbs,2)
    xj.requires_grad = True
    start = time.time()
    res, nit = torch_SGD(xj,room,room.objective_function, iters=2000, lr=0.03,momentum=0.9)
    end = time.time()
    
    SciPy_cg.append(room.J(res))
    SciPy_cg_nit.append(nit)
    SciPy_cg_time.append(end-start)
    room.show_plane(res)
    # print(f"Minima at\n{room.to_pos(res.x).round(2)}")
    # Fletcher Reeves

    room = Roof(args.L, args.B, args.H, objective_type=args.obj_function)

    start = time.time()
    x, count = SGD(x0, room.J, room.gradient,alpha=0.03, momentum=0.9, epsilon=1e-6, num_iters=2000)
    end = time.time()

    FR.append(float(room.J(x)))
    FR_nit.append(count)
    room.show_plane(x)

    print(f"\nMinimum Value = {room.J(x)}")
    print(f"\nMinima at\n{room.to_pos(x).round(2)}")
    FR_time.append(end-start)



if __name__ == '__main__':
    args = parser.parse_args()
    print('Loss:', SciPy_cg, FR)
    print('NIT:', SciPy_cg_nit, FR_nit)
    print('Time:',SciPy_cg_time, FR_time)
    import pickle as pkl
    for i in range(10):
        optimise(args,i)
    # pkl.dump([SciPy_cg, FR, SciPy_cg_nit, FR_nit, SciPy_cg_time, FR_time], open('data/nelder_mead/4bulbs.pkl','wb+'))


