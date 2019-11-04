from math import e

import numpy as np
import numpy.linalg as la
from scipy.special import expit

eps = np.float64(0.0001)


def get_elem(i, j):
    if i == j:
        return 1.0
    return 0.0


def get_e_k(n: int, k: int, tp=np.float64) -> np.ndarray:
    return np.array([get_elem(i, k) for i in range(n)]).reshape(-1, 1).astype(dtype=tp)


def generate_matrix(dim_x: int, dim_y: int, tp=np.float64) -> np.ndarray:
    m = np.random.rand(dim_x, dim_y) * 2 - np.ones((dim_x, dim_y), dtype=tp)
    return m


def getI(dim: int, lam) -> np.ndarray:
    return np.eye(dim) * lam


def get_min_eig(m: np.ndarray):
    eigenvalue, _ = la.eig(m)
    return min(eigenvalue)


def generate_pd_matrix(dim: int, tp=np.float64) -> np.ndarray:
    m = generate_matrix(dim, dim)
    m = (m + m.T) / 2
    min_eigen = get_min_eig(m)
    if min_eigen > 0:
        return m
    return getI(dim, tp(-min_eigen)) + getI(dim, eps) + m


def calc_gradient(func, x: np.ndarray, epsilon, tp=np.float64) -> np.ndarray:
    df = []
    for k in range(len(x)):
        dx = get_e_k(len(x), k).reshape(len(x)) * epsilon
        df.append((func(x + dx) - func(x - dx)) / (2 * epsilon))
    return np.array(df).astype(dtype=tp)


def calc_gessian(func, x: np.ndarray, epsilon, tp=np.float64) -> np.ndarray:
    return np.array(
        [calc_gradient(lambda y: calc_gradient(func, y, epsilon)[k], x, epsilon) for k in range(len(x))]
    ).reshape(len(x), len(x)).astype(dtype=tp)


def test_gradient(func, grad, dim, epsilon, tests, max_err):
    for i in range(tests):
        x = generate_matrix(dim, 1).reshape(dim)
        exp_grad = calc_gradient(func, x, epsilon)
        found_grad = grad(x)
        dist = np.max(abs(exp_grad - found_grad))
        if dist > max_err:
            print(i)
            print('Fail: ' + str(dist))
            return
    print('passed')


def test_gessian(func, gessian, dim, epsilon, tests, max_err):
    for i in range(tests):
        x = generate_matrix(dim, 1).reshape(dim)
        exp_gessian = calc_gessian(func, x, epsilon)
        found_gessian = gessian(x)
        dist = la.norm(exp_gessian - found_gessian)
        if dist > max_err:
            print(i)
            print('Fail: ' + str(dist))
            return
    print('passed')


# multipliction matrix on vector: number of elements in matrix operations
# scalar multiplication of two vectors: vector dimension operations
class FunctionHolder():
    def __init__(self, X, y):
        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.calls = 0
        self.X = X
        self.y = y
        self.operations = self.N * self.dim
        self.XTy = np.dot(X.T, y)
        self.xxT = np.array([np.dot(x.reshape(self.dim, 1), x.reshape(1, self.dim)) for x in X])

    def f(self, w):
        self.calls += 1
        self.operations += self.dim + self.dim * self.N + self.N * 3  # matrix multiplication, scalar multiplication, operations with vector
        return (-np.dot(w, self.XTy) + np.sum(np.log(1 + np.exp(np.dot(self.X, w))))) / self.N

    def grad(self, w):
        self.calls += 1
        ewx = expit(np.dot(self.X, w))
        return ((-np.dot(self.X.T, self.y) + np.dot(self.X.T, ewx)) / self.N).reshape(self.X.shape[1])

    def gessian(self, w):
        self.calls += 1
        Xw = np.dot(self.X, w)
        coefs = expit(Xw) / (1 + e ** Xw)

        self.operations += self.N ** 2 * self.dim

        return sum([self.xxT[i] * coefs[i] for i in range(self.N)]) / self.N