import math
import numpy as np

def f(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def gss(f, x, s, tol=1e-5, maxiter=50):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.
    """
    invphi = (math.sqrt(5)-1)/2
    invphi2 = invphi ** 2

    a, b = x - 1 * s,  x + 1 * s
    h = b - a
    if np.abs(h).max() <= tol:
        return (((a+b)/2 - x) @ s.T) / (s @ s.T)

    try:
        n = int((math.log(tol / np.abs(h).max()) / math.log(invphi)))
        n = min(n, maxiter)
    except:
        print(s)
        exit()

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        return (((a+d)/2 - x) @ s.T) / (s @ s.T)
    else:
        return (((c+b)/2 - x) @ s.T) / (s @ s.T)



if __name__ == "__main__":

    print(gss(f, np.array([2, 1]), np.array([2, 3])))

