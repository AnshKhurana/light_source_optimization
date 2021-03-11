import numpy as np
from math import sqrt

c=10
k=0
n=2
alpha=0.7
beta=0.5
gamma=1
delta= 0.2
e1=0.5
e2=0.5

xj= np.asarray([3,3])
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

evaluations=[objective_function(x) for x in simp]
sort_index=np.argsort(np.asarray(evaluations))
evaluations=[evaluations[x] for x in sort_index]
simp=[simp[x] for x in sort_index]
xw=simp[-1]
xl=simp[-2]
xb=simp[0]
fw=objective_function(xw)
fb=objective_function(xb)
fl=objective_function(xl)


# def shrink_simplex(simp):
# 	global xw,xl,xb
# 	simp= [xb + delta*(x-xb) for x in simp]
# 	evaluations=[objective_function(x) for x in simp]
# 	sort_index=np.argsort(np.asarray(evaluations))
# 	evaluations=[evaluations[x] for x in sort_index]
# 	simp=[simp[x] for x in sort_index]
# 	xw=simp[-1]
# 	xl=simp[-2]
# 	xb=simp[0]
# 	fw=objective_function(xw)
# 	fb=objective_function(xb)
# 	fl=objective_function(xl)

i=0

while (fw-fb) > 1e-10:

	evaluations=[objective_function(x) for x in simp]
	sort_index=np.argsort(np.asarray(evaluations))
	evaluations=[evaluations[x] for x in sort_index]
	simp=[simp[x] for x in sort_index]
	xw=simp[-1]
	xl=simp[-2]
	xb=simp[0]
	fw=objective_function(xw)
	fb=objective_function(xb)
	fl=objective_function(xl)

	i+=1
	xa= np.mean(np.asarray(simp[0:-1]))
	xr= xa + alpha*(xa-xw)

	print(xw,xl,xb,xr)
	fr=objective_function(xr)
	if fr<fb:
		xe= xa + gamma*(xr-xa)
		fe=objective_function(xe)
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
			fc=objective_function(xc)
			if fc<fw:
				simp[-1]=xc
				print("h4")
			else:
				simp= [xb + delta*(x-xb) for x in simp]
				print("simp1")
		else:
			xo= xa + beta*(xa-xw) 
			fo=objective_function(xo)
			if fo<=fr:
				simp[-1]=xo
				print("h5")
			else:
				simp= [xb + delta*(x-xb) for x in simp]
				print("simp2")


	# exit(0)






