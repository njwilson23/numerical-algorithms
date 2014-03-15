""" Provide root-finding schemes such as Newton's method, modified Newton's
method, fixed-point iteration. """

import numpy as np

def _newton_iters(f, J, u, tol=1e-8, maxiter=50):
    """ Newton iteration for scalar data. """
    it = 0
    fu = tol+1
    
    while (it < maxiter) and (abs(fu) > tol):
        fu = f(u)
        u = u - (fu / J(u))
        it += 1
    return u

def _newton_iterv(f, J, x0, tol=1e-8, maxiter=50):
    """ Newton iteration for vector data. """
    x = x0.copy()
    it = 0
    norm = tol + 1

    while (it < maxiter) and (norm > tol):
        x = x + np.linalg.solve(J(x), -f(x))
        norm = np.linalg.norm(f(x), ord=2)
        it += 1
    return x

def newton_iter(f, J, u, tol=1e-8, maxiter=50):
    """ Newton iteration. Dispatches to a method for either scaler or vector
    data in *u*.

    <f::Function, J::Function, u::Float[], tol::Float, maxiter::Int>
    """
    if hasattr(u, "__iter__"):
        return _newton_iterv(f, J, u, tol, maxiter)
    else:
        return _newton_iters(f, J, u, tol, maxiter)


