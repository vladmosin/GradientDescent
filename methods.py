import time
from enum import Enum

import numpy as np
import scipy.linalg as sla
import scipy.optimize as so
from numpy.linalg import norm as norm
from scipy.optimize import brentq


def multiply_gessian(f_holder, x0, v, eps=np.float64(1e-10)):
    grad = f_holder.grad
    return (grad(x0 + eps * v) - grad(x0)) / eps


def conjugate_gradient(f_holder, x0, v, eps=np.float64(1e-8)):
    not_oracle_operations = 0
    size = len(x0)
    multiply = lambda x, v: f_holder.gessian_at_point(x, v)

    r = v - multiply(x0, x0)
    d = np.zeros_like(r)
    x = np.copy(x0)
    t = 1
    stop_condition = (norm(v) * eps) ** 2
    multiply = lambda x, v: f_holder.gessian_at_point(x, v)

    not_oracle_operations += size

    while True:
        norm_r = norm(r) ** 2
        if norm_r <= stop_condition:
            break

        d = r - d * norm_r / t
        t = norm_r
        y = multiply(x0, d)
        a = t / d.dot(y)
        x += a * d
        r -= a * y

        not_oracle_operations += size * 9

    f_holder.operations += not_oracle_operations
    return x


def ternary_search_full(func, left, right, eps=np.float64(1e-7), fleft=None, fright=None):
    if right - left < eps:
        return (left + right) / 2, 0, 0

    phi = (3 - 5 ** 0.5) / 2
    fcalls = 0

    new_left, new_right = fleft, fright
    if new_left is None:
        new_left = left + (right - left) * phi
        fcalls += 1
    if new_right is None:
        new_right = right - (right - left) * phi
        fcalls += 1

    if func(new_left) > func(new_right):
        x, it, calls = ternary_search_full(func, new_left, right, eps, new_right, None)
        return x, it + 1, calls + fcalls
    else:
        x, it, calls = ternary_search_full(func, left, new_right, eps, None, new_left)
        return x, it + 1, calls + fcalls


def gen_poly(_pow):
    return np.random.normal(-1, 1, _pow + 1)


def ternary_search(func, left, right, eps=np.float64(1e-7)):
    x, _, _ = ternary_search_full(func, left, right, eps=eps)
    return x


def full_brent(grad):
    from_x, to_x = -1, 1
    while (grad(from_x) * grad(to_x) > 0):
        from_x *= 2
        to_x *= 2
    result = brentq(grad, from_x, to_x, full_output=True)
    alpha, f_calls = result[0], result[1].function_calls
    return alpha, f_calls


def brent(grad):
    alpha, _ = full_brent(grad)
    return alpha


def get_grad(poly):
    size = len(poly)
    grad = []
    for i in range(0, size - 1):
        grad.append(poly[i] * (size - i - 1))

    return np.array(grad)


def calc_poly(poly, x):
    ans = 0.0
    size = len(poly)
    for i in range(size - 1, -1, -1):
        ans = ans * x + poly[i]

    return ans


def compare_brent_ternary():
    from_x, to_x = np.float64(-1.0), np.float64(1.0)
    pows = [2, 4, 6, 8]
    polys = [gen_poly(_pow) for _pow in pows]
    grads = [get_grad(poly) for poly in polys]

    polys_f = [lambda x: calc_poly(poly, x) for poly in polys]
    grads_f = [lambda x: calc_poly(poly, x) for poly in grads]

    value_b, value_g, calls_b, calls_g = [], [], [], []
    for i in range(len(pows)):
        vg, _, cg = ternary_search_full(polys_f[i], from_x, to_x)
        vb, cb = full_brent(grads_f[i])

        # print(vb)
        # print(vg)

        value_b.append(polys_f[i](vb))
        value_g.append(polys_f[i](vg))
        calls_b.append(cb)
        calls_g.append(cg)

    print(value_b)
    print(value_g)
    print(calls_b)
    print(calls_g)


def armijo(f, grad, n=np.float64(1e-6), c=np.float64(0.1), tp=np.float64):
    alpha = 1.0
    f_0, grad_0 = f(0), grad(0)
    oracle_calls = 3

    while f(alpha) > f_0 + c * alpha * grad_0:
        alpha = alpha / 2
        oracle_calls += 1
    return alpha


def wolfe(f_holder, start_point, direction):
    result = so.line_search(f_holder.f, f_holder.grad, start_point, direction)
    alpha, oracle_calls = result[0], result[1] + result[2]
    return alpha


def lipschitz(f_holder, x0):
    L = np.float64(1)
    f_x0, grad_x0 = f_holder.f(x0), f_holder.grad(x0)
    while True:
        grad_x = f_holder.grad(x0)
        f_holder.operations += 2 * len(x0)
        if f_holder.f(x0 - grad_x / L) > f_x0 + 3 / 2 / L * norm(grad_x) ** 2:
            L *= 2
        else:
            break

    return 1 / L


class SearchType(Enum):
    ARMIJO = 1,
    WOLFE = 2,
    BRENT = 3,
    TERNARY = 4,
    LIPSCHITZ = 5


class Statistics:
    def __init__(self, times, xs, oracles, operations=None):
        self.times = times
        self.oracles = oracles
        self.xs = xs
        self.operations = operations


def gradient_descent(f_holder, x, search_type, epsilon=np.float64(1e-6), stop_criterion=np.float64(1e-6),
                     iterations=100000, tp=np.float64):
    f, grad = f_holder.f, f_holder.grad
    ngrad_x0 = norm(grad(x))
    from_alp, to_alp = np.float64(0), np.float64(100)
    x = np.copy(x)
    start_time = time.time()

    tot_oracle_calls = 0
    times, xs, oracles, operations = [0], [np.copy(x)], [0], [0]

    for i in range(iterations):
        grad_x = grad(x)
        func = lambda alp: f(x - grad_x * alp)
        grad_func = lambda y: grad_x.dot(grad(x - y * grad_x))

        if search_type == SearchType.ARMIJO:
            alpha = armijo(func, grad_func, epsilon, tp=tp)
        elif search_type == SearchType.WOLFE:
            alpha = wolfe(f_holder, x, -grad_x)
        elif search_type == SearchType.BRENT:
            alpha = brent(lambda y: grad_x.dot(grad(x - y * grad_x)))
        elif search_type == SearchType.TERNARY:
            alpha = ternary_search(func, from_alp, to_alp)
        else:
            alpha = lipschitz(f_holder, x)
        x -= grad_x * alpha
        tot_oracle_calls = f_holder.calls

        current_time = time.time()

        oracles.append(tot_oracle_calls)
        xs.append(np.copy(x))
        times.append(current_time - start_time)
        operations.append(f_holder.operations)

        if norm(grad(x)) ** 2 <= ngrad_x0 ** 2 * stop_criterion:
            return x, Statistics(times=times, oracles=oracles, xs=xs, operations=operations)

    return x, Statistics(times=times, oracles=oracles, xs=xs, operations=operations)


EPS = np.float64(1e-6)


def newton(f_holder, x, iterations=10000, stop_criterion=np.float64(1e-6), eps=None):
    f, grad, gessian = f_holder.f, f_holder.grad, f_holder.gessian
    size = len(x)
    start_time = time.time()
    oracles, times, xs, operations = [0], [0], [np.copy(x)], [0]

    ngrad_x0 = norm(grad(x))
    for i in range(iterations):
        grad_x = grad(x)
        if eps is not None:
            x -= conjugate_gradient(f_holder, x, grad_x, eps=eps(norm(grad(x))))
        else:
            E_eps = EPS * np.eye(size)
            x -= sla.cho_solve(sla.cho_factor(gessian(x) + E_eps), grad(x)).reshape(size)

        f_holder.operations += x.shape[0] ** 2
        current_time = time.time()
        oracles.append(f_holder.calls)
        times.append(current_time - start_time)
        xs.append(np.copy(x))
        operations.append(f_holder.operations)

        if norm(grad(x)) ** 2 <= ngrad_x0 ** 2 * stop_criterion:
            return x, Statistics(times=times, oracles=oracles, xs=xs, operations=operations)

    return x, Statistics(times=times, oracles=oracles, xs=xs, operations=operations)