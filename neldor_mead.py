import numpy as np
from math import sqrt
from environment import Room
import copy

num_bulbs=2

k=0
n=3* num_bulbs

c=20
alpha=1
beta=2
gamma=-0.5
delta= 0.5
e1=1e-3
e2=1e-3

def nelder_mead(f, x_start,
                step=3, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=100,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:',best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres




# xj =np.zeros(n)

# def calculate_intensity(bulbs, L = 5, res=10):
#     y, x = torch.meshgrid(torch.linspace(0,15,15*res), torch.linspace(0,25,25*res))
#     I = torch.zeros(15*res,25*res)
#     for b in bulbs:
#         I += 1 / ((b[1]-x)**2+(b[0]-y)**2+ (2.5)**2)
#     return I

def objective_function(x):
	if x[0]<1.0 or x[1]<1.0:
		return np.dot(x,x) +20
	else:
		return np.dot(x,x)

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


room = Room(10, 10, 10, plane_height=5, objective_type='simple_min')
xj = (np.array([[7,7,7],[5,5,7]])).reshape(-1)

def func(xw):
	# print(xw)
	return room.objective_function(xw.reshape(num_bulbs,3))
	# return np.dot(xw,xw)

print(nelder_mead(func,xj))

exit(0)


def simplex(x0):
	lis=[x0]
	for i in range(n):
		x=np.empty(n)
		b=c*(sqrt(n+1)-1)/(n*sqrt(2))
		a=b + c/sqrt(2)
		x.fill(b)
		print(b,a)
		x[i]=a
		x+=xj
		lis.append(x)
	return lis

simp=simplex(xj)
print(simp)

evaluations=[room.objective_function(x.reshape(num_bulbs,3)) for x in simp]
sort_index=np.argsort(np.asarray(evaluations))
evaluations=[evaluations[x] for x in sort_index]
simp=[simp[x] for x in sort_index]
xw=simp[-1]
xl=simp[-2]
xb=simp[0]
fw=room.objective_function(xw.reshape(num_bulbs,3))
fb=room.objective_function(xb.reshape(num_bulbs,3))
fl=room.objective_function(xl.reshape(num_bulbs,3))


i=0

while (fw-fb) > 1e-6:

	evaluations=[room.objective_function(x.reshape(num_bulbs,3)) for x in simp]
	sort_index=np.argsort(np.asarray(evaluations))
	evaluations=[evaluations[x] for x in sort_index]
	simp=[simp[x] for x in sort_index]
	xw=simp[-1]
	xl=simp[-2]
	xb=simp[0]
	fw=room.objective_function(xw.reshape(num_bulbs,3))
	fb=room.objective_function(xb.reshape(num_bulbs,3))
	fl=room.objective_function(xl.reshape(num_bulbs,3))

	i+=1
	xa= np.mean(np.asarray(simp[0:-1]))
	xr= xa + alpha*(xa-xw)

	print(xb, fb)
	fr=room.objective_function(xr.reshape(num_bulbs,3))
	if fr<fb:
		xe= xa + gamma*(xr-xa)
		fe=room.objective_function(xe.reshape(num_bulbs,3))
		if fe<fb:
			simp[-1]=xe
			print("h1")
		else:
			simp[-1]=xr
			print("h2")
	elif fr<=fl:
		simp[-1]=xr
		print("h3")
	else:
		if fr>fw:
			xc= xa - beta*(xa-xw)
			fc=room.objective_function(xc.reshape(num_bulbs,3))
			if fc<fw:
				simp[-1]=xc
				print("h4")
			else:
				simp= [xb + delta*(x-xb) for x in simp]
				print("simp1")
		else:
			xo= xa + beta*(xa-xw) 
			fo=room.objective_function(xo.reshape(num_bulbs,3))
			if fo<=fr:
				simp[-1]=xo
				print("h5")
			else:
				simp= [xb + delta*(x-xb) for x in simp]
				print("simp2")


	# exit(0)






