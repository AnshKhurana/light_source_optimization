import jax.numpy as np
from jax import grad, hessian
from numpy import random


def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def newtons_method(grad, hessian, x, s, n):
    """
    Line Search using Modified Newton's Method
    """
    α = 0
    for _ in range(n):
        G = grad(x)
        T = hessian(x) @ s
        dα = -(G @ T.T) / (T @ T.T)
        α += dα
        x += dα * s
    return α


def fletcher_reeves(J, grad, hessian, X_0, n_iter=1, ε_a=1e-6,
                    ε_r=1e-6, ε_g=1e-4, verbose=False):
    """
    Fletcher Reeves Algorithm
    """
    X = X_0
    k = 0
    while True:
        G = grad(X)
        if verbose:
            print(k, X)
            print(np.linalg.norm(G, ord=float('inf')))
        if k == 0:
            S = -G / np.linalg.norm(G, ord=float('inf'))
        else:
            β = G.T @ (G-G_old) / (G_old.T @ G_old)
            S = -G / np.linalg.norm(G, ord=float('inf')) + β*S_old
        X_old, G_old, S_old = X, G, S
        X += newtons_method(grad, hessian, X, S, n_iter) * S
        k += 1
        J_old = J(X_old)
        if abs(J(X)-J_old) <= ε_a + ε_r*abs(J_old)\
                and np.linalg.norm(G, ord=float('inf')) <= ε_g:
            break
        if np.abs(X-X_old).max() < 1e-6:
            break
    return X, k


if __name__ == "__main__":
    X_0 = np.array([2.0, 2.0, 0.5])
    X = fletcher_reeves(rosen,
                        grad(rosen),
                        hessian(rosen),
                        X_0, ε_a=1e-5,
                        ε_r=1e-5, ε_g=1e-4,
                        verbose=True)
    print(X)
