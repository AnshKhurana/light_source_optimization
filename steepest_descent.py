"""Implementation of steepest descent algorithm."""
import numpy as np


def test_obj(X):
    return np.linalg.norm(X - np.array([1.0, 2.0, 3.0]))**2


def test_grad(X):
    return 2*(X - np.array([1.0, 2.0, 3.0]))


def steepest_descent(X_0, alpha_0, obj_fn=test_obj, grad_fn=test_grad,
                     num_itr=128, eps_a=1e-3, eps_r=1e-3, eps_g=1e-3):

    X = X_0
    prev_G = grad_fn(X)
    prev_alpha = alpha_0
    prev_J = obj_fn(X)

    for i in range(num_itr):

        print("Itr", i, "X", X, "obj function", obj_fn(X), "gradient", grad_fn(X))

        J = obj_fn(X)
        G = grad_fn(X)
        S = -G / (np.linalg.norm(G))

        # if np.abs(J - prev_J) > eps_a + eps_r * np.abs(prev_J) or np.linalg.norm(G) > eps_g:
        #     print("Converged!")
        #     break
        if np.linalg.norm(G) < eps_g:
            print("converged")
            break

        alpha = prev_alpha * (prev_G.T @ prev_G)/(G.T @ G)
        X += alpha*S
        prev_alpha = alpha
        prev_G = G
        prev_J = J

    return X


if __name__ == "__main__":

    X0 = np.array([1, 1, 1])
    X_0 = np.array([0.0, 0.0, 0.0])
    alpha_0 = 1e-2 * np.ones(3)
    print(np.linalg.norm(X0)**2)
    steepest_descent(X_0, alpha_0, num_itr=2000)