import jax.numpy as np
from jax import grad


def test_obj(X):
    return np.linalg.norm(X - np.array([1.0, 2.0, 3.0]))**2


def test_grad(X):
    return 2*(X - np.array([1.0, 2.0, 3.0]))


def test_hessian(X):
    return 2 * np.eye(3)


def newtons_method(grad, hessian, X_0, S, ε=1e-5):
    X = X_0
    k = 0
    α = 0
    while k < 10:
        T = hessian(X) @ S
        G = grad(X)
        α += -(G @ T.T) / (T @ T.T) / 2
        if np.linalg.norm(α*S, 2) < ε:
            break
        X = X + α * S
        k += 1
    return α


def fletcher_reeves(J, grad, X_0=np.zeros(3), ε_a=1e-5, ε_r=1e-5, ε_g=1e-5):
    X = X_0
    k = 1
    while True:
        G = grad(X)
        if k == 1:
            S = -G / np.linalg.norm(G, 2)
        else:
            β = G.T @ G / (G_old.T @ G_old)
            S = -G / np.linalg.norm(G, 2) + β*S_old
        G_old, S_old = G, S
        α = newtons_method(test_grad, test_hessian, X, S)
        X_old = X
        X += α * S
        k += 1
        if abs(J(X)-J(X_old)) <= ε_a + ε_r*abs(J(X_old))\
                and np.linalg.norm(G, 2) <= ε_g:
            break
    return X


if __name__ == "__main__":
    X = fletcher_reeves(test_obj, test_grad)
    print(X)
