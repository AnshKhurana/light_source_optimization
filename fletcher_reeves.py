import jax.numpy as np
from jax import grad, hessian


def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def newtons_method(grad, hessian, x, s):
    """
    Line Search using Modified Newton's Method
    """
    α = 0
    for _ in range(1):
        G = grad(x)
        T = hessian(x) @ s
        dα = -(G @ T.T) / (T @ T.T)
        α += dα
        x += dα * s
    return α

def fletcher_reeves(J, grad, hessian, X_0=np.zeros(3), ε_a=1e-5, ε_r=1e-5, ε_g=1e-4):
    """
    Fletcher Reeves Algorithm
    """
    X = X_0
    k = 0
    while True:
        G = grad(X)
        if k == 0:
            S = -G / np.linalg.norm(G)
        else:
            β = G.T @ G / (G_old.T @ G_old)
            S = -G / np.linalg.norm(G) + β*S_old
        X_old, G_old, S_old = X, G, S
        X += newtons_method(grad, hessian, X, S) * S
        k += 1
        if abs(J(X)-J(X_old)) <= ε_a + ε_r*abs(J(X_old))\
                and np.linalg.norm(G) <= ε_g:
            break
    return X


if __name__ == "__main__":
    X_0 = np.array([2.0,2.0, 0.5])
    X = fletcher_reeves(rosen, grad(rosen), hessian(rosen), X_0)
    print(X)
