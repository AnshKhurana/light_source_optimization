import jax.numpy as np
from jax import grad, hessian


def test_obj(X):
    return np.linalg.norm(X - np.array([1.0, 2.0, 3.0]))**2


def fletcher_reeves(J, grad, hessian, X_0=np.zeros(3), ε_a=1e-5, ε_r=1e-5, ε_g=1e-5):
    """
    Fletcher Reeves Algorithm
    """

    def newtons_method(X_0, S, ε=1e-6):
        """
        Line Search using Modified Newton's Method
        """
        X = X_0
        k = 0
        α = 0
        while k < 25:
            T = hessian(X) @ S
            G = grad(X)
            dα = -(G @ T.T) / (T @ T.T) / 2
            if np.linalg.norm(dα*S, 2) < ε:
                break
            α += dα
            X += dα * S
            k += 1
        return α

    X = X_0
    k = 0
    while True:
        G = grad(X)
        if k == 0:
            S = -G / np.linalg.norm(G, 2)
        else:
            β = G.T @ G / (G_old.T @ G_old)
            S = -G / np.linalg.norm(G, 2) + β*S_old
        G_old, S_old = G, S
        α = newtons_method(X, S)
        X_old = X
        X += α * S
        k += 1
        if abs(J(X)-J(X_old)) <= ε_a + ε_r*abs(J(X_old))\
                and np.linalg.norm(G, 2) <= ε_g:
            break
    return X


if __name__ == "__main__":
    X = fletcher_reeves(test_obj, grad(test_obj), hessian(test_obj))
    print(X)
