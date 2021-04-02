import numpy as np
from math import sqrt
from environment import Roof
import copy
from scipy.optimize import minimize


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
        xa= np.mean(np.asarray(simp[0:-1]),axis=0) #centroid
        xr= xa + alpha*(xa-xw) #reflection
        fr=func(xr)
        if fb<=fr < fl:
            simp[-1]=xr  #accept reflection
        elif fr< fb:
            xe= xa + gamma*(xa-xw)  #expansion
            fe=func(xe)     
            if fe<fr: 
                simp[-1]=xe        #accept expansion
            else:
                simp[-1]=xr        #accept reflection
        else:
            xc= xa + beta*(xa-xw)
            fc=func(xc)            #contraction
            if fc<fw:
                simp[-1]=xc        #accept contraction
            else:
                simp= [xb + delta*(x-xb) for x in simp] #shrink the simplex
    return xb,fb


if __name__ == "__main__":

    num_bulbs=3
    np.random.seed(42)
    xj = np.random.randn(2 * num_bulbs)*2

    room = Roof(20, 10, 7, plane_height=5, objective_type='simple_std')
    
    room.show_plane(xj)
    print(room.objective_function(xj))
    # def f(xw):
    # 	return room.objective_function(xw)

    ans = nelder_mead(room.objective_function, xj)
    # ans = minimize(room.objective_function, xj, method='Nelder-Mead')
    print(ans)
    room.show_plane(ans[0])



