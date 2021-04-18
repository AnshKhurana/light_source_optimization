import jax.numpy as np
import numpy as onp
from math import sqrt
from environment import Roof
import copy
from scipy.optimize import minimize
from jax import grad

import signal
import sys

num_bulbs=5
onp.random.seed(42)
xj = onp.random.randn(2 * num_bulbs)*2
room = Roof(20, 10, 7, plane_height=5, objective_type='simple_std')


def signal_handler(sig, frame):
	room.show_plane(xj)
	print('You pressed Ctrl+C!')
	sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')



def rosen(X):
	"""
	This R^2 -> R^1 function should be compatible with algopy.
	http://en.wikipedia.org/wiki/Rosenbrock_function
	A generalized implementation is available
	as the scipy.optimize.rosen function
	"""
	x = X[0]
	y = X[1]
	a = 1. - x
	b = y - x*x
	return a*a + b*b*100.


def SGD(xj, func, gradient,alpha=0.01, momentum=0.9, epsilon=1e-6):
	# global xj
	velocity=np.zeros_like(xj)
	r1=gradient(xj)
	print(r1,xj)
	i=0
	while np.linalg.norm(r1) > epsilon:
		if i%100==0:
			print(func(xj),np.linalg.norm(r1))
			alpha-=0.001
		if alpha<0:
			break
		velocity = momentum*velocity -alpha*r1
		xj= xj + velocity
		# xj-=alpha*r1
		r1=gradient(xj)
		i+=1
	return xj, i



if __name__ == "__main__":	
	print(room.objective_function(xj))
	ans, _ = SGD(xj, room.objective_function, room.gradient)
	room.show_plane(ans)



