""" Collection of classes that can be composed to contruct numerical methods
for solving the wave equation u_tt = c^2 u_xx.

Possibly abuses OO conventions.    
"""

import numpy as np

def construct_semidiscrete(*args):
    """ Construct a semidiscrete method by composing a difference scheme and
    some number of callables (e.g. boundary conditions, limiters). """
    A = args[0]()
    for f in args[1:]:
        A = f(A)
    return A

class CenteredSpaceDifference1(object):

    def __init__(self, n, k, h, c):
        """ Implements a centered (symmetric) space difference, i.e.
    
            u_1 - 2u_0 + u_-1

        Parameters:
        -----------
        n       domain size
        k       time step
        h       space step
        c       wave speed

        <n::int, k::float, h::float, c::float>
        """
        self.n = n
        self.k = k
        self.h = h
        self.c = c
        return

    def __call__(self):
        """ Return the difference matrix. """
        I = np.eye(self.n)
        R = np.diag(np.ones(self.n-1), 1)
        r2 = self.k**2 / self.h**2
        A = self.c**2 * (r2*R + 2*(1-r2)*I + r2*R.T)
        return A

class PeriodicBoundary1(object):

    def __init__(self):
        """ Applies a periodic boundary condition to a one-dimensional
        difference matrix.
        
        Limitations:
        ------------
        assumes a symmetric stencil with width 3
        """
        return

    def __call__(self, A):
        m, n = A.shape[:2]
        if (m < 3) or (n < 3):
            raise ValueError("'A' must be at least 3x3")
        A[0,-1] = A[1,0]
        A[-1,0] = A[-2,-1]
        return A

