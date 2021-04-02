"""Implementation of optimization algorithms."""
import jax.numpy as np
import numpy as np
import scipy.optimize as sopt

def test_obj(X):
    return np.linalg.norm(X - np.array([1.0, 2.0, 3.0]))**2

def test_grad(X):
    return 2*(X - np.array([1.0, 2.0, 3.0]))

def steepest_descent():
    X_0 = np.array([10.0, -3.0, 0.0])
    num_itr = 128
    X = X_0
    alpha_0 = 1e-2 * np.ones(3)
    prev_G = test_grad(X)
    prev_alpha = alpha_0
    prev_J = test_obj(X)
    eps_a, eps_r, eps_g = 1e-3, 1e-3, 1e-3

    for i in range(num_itr):

        print("Itr", i, "X", X, "obj function", test_obj(X), "gradient", test_grad(X))

        J = test_obj(X)
        G = test_grad(X)
        S = -G/(np.linalg.norm(G))

        # if np.abs(J - prev_J) > eps_a + eps_r * np.abs(prev_J) or np.linalg.norm(G) > eps_g:
        #     print("Converged!")
        #     break
        def f1d(alpha):
            return test_obj(X + alpha*S)
        alpha = sopt.golden(f1d)
        print("Alpha: ", alpha)
        # alpha = prev_alpha * (prev_G.T @ prev_G)/(G.T @ G)
        X += alpha*S
        prev_alpha = alpha
        prev_G = G
        prev_J = J

    pass

def nelder_mead():
    pass

def conjugate_gradient():
    pass



if __name__ == "__main__":

    X = np.array([1, 1, 1])
    print(np.linalg.norm(X)**2)
    steepest_descent()