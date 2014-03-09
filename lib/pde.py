""" Collection of classes that can be composed to contruct numerical methods
for solving the wave equation u_tt = c^2 u_xx.

Possibly abuses OO conventions.    
"""

import numpy as np
import scipy.sparse as sp

class DifferenceSchemeBase(object):

    def __init__(self):
        self.items = []
        
    def append(self, other):
        self.items.append(other)
        return self

class CenteredDifferenceScheme1(DifferenceSchemeBase):

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
        super(type(self), self).__init__()
        self.n = n
        self.k = k
        self.h = h
        self.c = c
        return

    def matrix(self, rhs_u=0):
        """ Return the difference matrix for the explicit discrete scheme.
        
        *rhs_u* indicates the multiple of the current timestep to add to the RHS
        """
        I = np.eye(self.n)
        R = np.diag(np.ones(self.n-1), 1)
        r2 = self.k**2 / self.h**2
        A = self.c**2 * (r2*R  -2*r2*I + r2*R.T + rhs_u*I)

        for cls in self.items:
            A = cls(A)
        return A

class CenteredDifferenceScheme2(DifferenceSchemeBase):

    def __init__(self, shape, k, DX, c):
        """ Implements a centered (symmetric) space difference in two
        dimensions.

        <shape::int[], k::float, DX::float[], c::float>
        """
        super(type(self), self).__init__()
        self.shape = shape
        self.k = k
        self.DX = DX
        self.c = c
        return

    def matrix(self, rhs_u):
        """ Return the difference matrix *L* for the explicit discrete scheme.
        
        The differences in *u* can then be computed as

        `np.dot(L, u.ravel())`

        *rhs_u* indicates the multiple of the current timestep to add to the RHS
        """
        nx, ny = self.shape
        rx = self.k / self.DX[0]
        ry = self.k / self.DX[1]

        ex = np.ones(nx)
        ey = np.ones(ny)
        dxx = sp.diags([ex[1:], -2*ex, ex[:-1]], [-1, 0, 1], format='lil') * rx**2
        dyy = sp.diags([ey[1:], -2*ey, ey[:-1]], [-1, 0, 1], format='lil') * ry**2

        dxx += sp.diags([rhs_u*ex], [0], format='lil')
        dyy += sp.diags([rhs_u*ey], [0], format='lil')

        L = sp.kron(dxx, sp.eye(ny)) + \
            sp.kron(dyy, sp.eye(nx))

        for cls in self.items:
            L = cls(L.tolil(), self)

        return L.tocsc()


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

class PeriodicBoundary2(object):

    def __init__(self):
        """ Applies a periodic boundary condition to a one-dimensional
        difference matrix.
        
        Limitations:
        ------------
        assumes a symmetric stencil with width 3
        """
        return

    def __call__(self, L, schm):
        nx, ny = schm.shape
        for i in range(ny):
            for j in range(ny):
                L[i*nx, j*nx + nx-1] = 1.0
                L[i*nx + nx-1, j*nx] = 1.0
        return L

