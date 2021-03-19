import numpy as np
from math import sqrt
from environment import Room
import copy


def nelder_mead(func, xj, c=1, epsilon=10e-6, alpha=1., gamma=2., beta=-0.5, delta=0.5):
	n=len(xj)
	def simplex(x0):
		lis=[x0]
		for i in range(n):
			x=copy.copy(x0)
			x[i]+=c
			lis.append(x)
		return lis
	simp=simplex(xj)
	print(simp)
	evaluations=[func(x) for x in simp]
	sort_index=np.argsort(np.asarray(evaluations))
	evaluations=[evaluations[x] for x in sort_index]
	simp=[simp[x] for x in sort_index]
	xw=simp[-1]
	xl=simp[-2]
	xb=simp[0]
	fw=func(xw)
	fb=func(xb)
	fl=func(xl)
	i=0
	while (fw-fb) > epsilon:
		print("Iter No. ",i,xb, fb)
		i+=1
		evaluations=[func(x) for x in simp]
		sort_index=np.argsort(np.asarray(evaluations))
		evaluations=[evaluations[x] for x in sort_index]
		simp=[simp[x] for x in sort_index]
		xw=simp[-1]
		xl=simp[-2]
		xb=simp[0]
		fw=func(xw)
		fb=func(xb)
		fl=func(xl)
		xa= np.mean(np.asarray(simp[0:-1]),axis=0)
		xr= xa + alpha*(xa-xw)

		fr=func(xr)
		if fb<=fr < fl:
			simp[-1]=xr
		elif fr< fb:
			xe= xa + gamma*(xa-xw)
			fe=func(xe)
			if fe<fr:
				simp[-1]=xe
			else:
				simp[-1]=xr
		else:
			xc= xa + beta*(xa-xw)
			fc=func(xc)
			if fc<fw:
				simp[-1]=xc
			else:
				simp= [xb + delta*(x-xb) for x in simp]
	return xb,fb


if __name__ == "__main__":

	num_bulbs=1
	n=3* num_bulbs
	room = Room(20, 10, 7, plane_height=5, objective_type='simple_min')
	xj = (np.array([3,7,7])).reshape(-1)
	def f(xw):
		return room.objective_function(xw)
		
	print(nelder_mead(f,xj))



