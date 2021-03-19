import numpy as np
from math import sqrt
from environment import Room
import copy

num_bulbs=1

k=0
n=3* num_bulbs

c=1
alpha=1
beta=-0.5
gamma=2
delta= 0.5
e1=1e-3
e2=1e-3

room = Room(10, 15, 10, plane_height=5, objective_type='simple_min')
xj = (np.array([3,7,7])).reshape(-1)

def objective_function(x):
	if x[0]<1.0 or x[1]<1.0:
		return np.dot(x,x) +20
	else:
		return np.dot(x,x)

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)




def func(xw):
	return room.objective_function(xw)



def simplex(x0):
	lis=[x0]
	for i in range(n):
		x=copy.copy(x0)
		x[i]+=c
		lis.append(x)
	return lis

simp=simplex(xj)
print(simp)

evaluations=[room.objective_function(x) for x in simp]
sort_index=np.argsort(np.asarray(evaluations))
evaluations=[evaluations[x] for x in sort_index]
simp=[simp[x] for x in sort_index]
xw=simp[-1]
xl=simp[-2]
xb=simp[0]
fw=room.objective_function(xw)
fb=room.objective_function(xb)
fl=room.objective_function(xl)

while (fw-fb) > 1e-10:
	evaluations=[room.objective_function(x) for x in simp]
	sort_index=np.argsort(np.asarray(evaluations))
	evaluations=[evaluations[x] for x in sort_index]
	simp=[simp[x] for x in sort_index]
	xw=simp[-1]
	xl=simp[-2]
	xb=simp[0]
	fw=room.objective_function(xw)
	fb=room.objective_function(xb)
	fl=room.objective_function(xl)
	xa= np.mean(np.asarray(simp[0:-1]),axis=0)
	print("xa= ",xa,simp[0:-1])
	xr= xa + alpha*(xa-xw)

	print(xb, fb)
	fr=room.objective_function(xr)
	if fb<=fr < fl:
		simp[-1]=xr
		print("h2")

	elif fr< fb:
		xe= xa + gamma*(xa-xw)
		fe=room.objective_function(xe)
		if fe<fr:
			simp[-1]=xe
			print("h1")
		else:
			simp[-1]=xr
			print("h2")
	else:
		xc= xa + beta*(xa-xw)
		fc=room.objective_function(xc)
		if fc<fw:
			simp[-1]=xc
			print("h4")
		else:
			simp= [xb + delta*(x-xb) for x in simp]
			print("simp1")






