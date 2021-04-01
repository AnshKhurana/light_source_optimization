import jax.numpy as np
# import numpy as np
from jax import grad, hessian
import scipy.optimize as sopt
from golden_search import gss

def test_obj(X):
    return np.linalg.norm(X - np.array([1.0, 2.0, 3.0]))**2

def test_grad(X):
    return 2*(X - np.array([1.0, 2.0, 3.0]))

def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def bfgs(obj, grad, X_0, eps_a=1e-3, eps_r=1e-3, eps_g=1e-4, num_itr=500):

    X = X_0
    # B_inv_prev = 2 * np.eye(3)
    H = hessian(rosen)
    B_inv_prev = H(X)
    print(B_inv_prev)
    # B_prev = None
    G = grad(X)

    for i in range(num_itr):

        print("Itr", i, "X", X, "obj function", obj(X), "gradient", G)

        if np.linalg.norm(G) < eps_g:
            print("converged")
            break

        p = -(B_inv_prev @ G)
        # alpha = sopt.golden(lambda t: obj(X + t*p))
        alpha = gss(obj, X, p)
        print(alpha)
        # alpha, _, _ = strongwolfe(obj, grad, p, X, obj(X), grad(X))
        s = alpha * p
        X_next = X + s

        G_next = grad(X_next)
        y = G_next - G
        sy = s.T @ y
        # print(sy)
        second = ((sy + y.T @ B_inv_prev @ y)/(sy*sy))*(s @ s.T)
        third = ((B_inv_prev @ y @ s.T) + (s @ (y.T @ B_inv_prev)))/sy
        B_inv = B_inv_prev + second - third

        B_inv_prev = B_inv
        X = X_next
        G = G_next


def strongwolfe(myFx, grad,d,x0,fx0,gx0):

    maxIter = 3
    alpham = 20
    alphap = 0
    c1 = 1e-4
    c2 = 0.9
    alphax = 1
    gx0 = gx0.T @ d
    fxp = fx0
    gxp = gx0
    i = 1

    while True:
        xx = x0 + alphax*d
        # [fxx,gxx] = myFx(xx)
        fxx = myFx(xx)
        gxx = grad(xx)
        fs = fxx
        gs = gxx
        gxx = (gxx)@d
        if (fxx > fx0 + c1*alphax*gx0) or ((i > 1) and (fxx >= fxp)):
            [alphas,fs,gs] = Zoom(myFx, grad, x0,d,alphap,alphax,fx0,gx0);
            return [alphas,fs,gs]
        if abs(gxx) <= -c2*gx0:
            alphas = alphax
            return [alphas,fs,gs]
        if gxx >= 0:
            [alphas,fs,gs] = Zoom(myFx, grad, x0,d,alphax,alphap,fx0,gx0)
            return [alphas,fs,gs]

        alphap = alphax
        fxp = fxx
        gxp = gxx

        if i > maxIter:
            alphas = alphax
            return [alphas,fs,gs]
        #r = rand(1);%randomly choose alphax from interval (alphap,alpham)
        r = 0.8
        alphax = alphax + (alpham-alphax)*r
        i = i+1


def Zoom(myFx, grad, x0,d,alphal,alphah,fx0,gx0):
    c1 = 1e-4
    c2 = 0.9
    i = 0
    maxIter = 5
    while True:
        alphax = 0.5*(alphal+alphah)
        alphas = alphax
        xx = x0 + alphax*d
        # [fxx,gxx] = myFx(xx)
        fxx = myFx(xx)
        gxx = grad(xx)
        fs = fxx
        gs = gxx
        gxx = (gxx)@d
        xl = x0 + alphal*d
        fxl = myFx(xl)
        if (fxx > fx0 + c1 * alphax * gx0) or (fxx >= fxl):
            alphah = alphax
        else:
            if abs(gxx) <= -c2*gx0:
                alphas = alphax
                return [alphas, fs, gs]
            if gxx*(alphah-alphal) >= 0:
                alphah = alphal
            alphal = alphax
            i = i+1
            if i > maxIter:
                alphas = alphax
                return [alphas, fs, gs]


if __name__ == "__main__":

    # X = np.array([1, 1, 1])
    # print(np.linalg.norm(X)**2)
    bfgs(rosen, grad(rosen), np.array([0.0, 0.0]))


# X_0 = np.array([10.0, -3.0, 0.0])
# num_itr = 10
# alpha_0 = 1e-2 * np.ones(3)
# prev_G = test_grad(X)
# prev_alpha = alpha_0
# prev_J = test_obj(X)
# eps_a, eps_r, eps_g = 1e-3, 1e-3, 1e-3


 # if np.abs(J - prev_J) > eps_a + eps_r * np.abs(prev_J) or np.linalg.norm(G) > eps_g:
#     print("Converged!")
#     break
# def f1d(alpha):
#     return test_obj(X + alpha*S)
# alpha = sopt.golden(f1d)
# print("Alpha: ", alpha)
# alpha = prev_alpha * (prev_G.T @ prev_G)/(G.T @ G)
# X += alpha*S
# prev_alpha = alpha
# prev_G = G
# prev_J = J

# J = test_obj(X)
# G = grad(X)
# S = -G/(np.linalg.norm(G))