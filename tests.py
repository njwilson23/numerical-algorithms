
import unittest
import numpy as np
import sys

sys.path.append('lib')
import roots
import ode
import pde
import typecheck

def isnear(a, b, tol=1e-4):
    return abs(a-b) < tol

class RootFindTests(unittest.TestCase):

    def test_newton_scalar(self):
        """ Test scalar Newton iteration """
        f = lambda x: (x - 3)**2
        J = lambda x: 2 * (x - 3)
        res = roots._newton_iters(f, J, 0.0)
        # Analytical solution is 3
        self.assertTrue(isnear(res, 3))
        return res

    def test_newton_vector(self):
        """ Test vectorized Newton iteration with two uncoupled problems """
        f = lambda X: np.asarray(((X[0] - 3)**2,   (X[1] + 7)**3))
        J = lambda X: np.asarray(([2 * (X[0] - 3), 0.0],
                                  [0.0, 3 *  (X[1] + 7)**2]))
        res = roots._newton_iterv(f, J, np.asarray((0.0, 0.0)), tol=1e-12)
        # Analytical solution is [3, -7]
        self.assertTrue(isnear(res[0], 3))
        self.assertTrue(isnear(res[1], -7))
        return res


class CenteredDifference1_FD_Tests(unittest.TestCase):

    def setUp(self):
        # initial conditions
        def f(x):
           return np.clip(1.0 - 20.0 * x**2, 0.0, np.inf)**2
        
        def g(x):
           d = x[1] - x[0]
           return -np.r_[np.diff(f(x)), 0.0] / (2.0 * d)
        
        n = 101
        k = 1.6e-2
        h = 2.0 / (n+1)
        c = 1.0
        x = np.linspace(-1, 1, n+2)[1:-1]
        T = np.arange(0, 1.0+k/2, k)
        self.params = dict(f=f, g=g, n=n, k=k, h=h, c=c, x=x, T=T)
        return

    def test_leapfrog1d(self):
        """ test 1D CenteredDifferenceScheme1 using leapfrog time
        discretization on the wave equation with homogeneous Dirichlet
        boundary conditions """

        p = self.params
        scheme = pde.CenteredDifferenceScheme1(p['n'], p['h'])
        A = scheme.matrix() * p['k']**2 * p['c']
        
        x = p['x']
        T = p['T']

        U = np.empty((len(T), A.shape[0]))

        U[0] = p['f'](x)
        U[1] = p['f'](x) + p['g'](x) * p['k']
        
        i = 2
        for t in T[2:]:
            U[i] = np.dot(A, U[i-1]) - U[i-2] + 2*U[i-1]
            i += 1

        correctresult = np.array([
            2.40147757e-02,   5.37989269e-02,   7.74743496e-02,
             9.28869917e-02,   1.07627539e-01,   1.18849457e-01,
             1.22732437e-01,   1.13384163e-01,   8.64768915e-02,
             5.45747395e-02,   2.52829521e-02,   1.88927308e-03,
            -7.07555989e-03,  -6.56845809e-03,  -2.82590756e-04,
             4.59175559e-03,   5.82086855e-04,  -1.92335527e-03,
            -1.21584035e-03,   1.79549404e-03,   6.99946407e-04,
            -2.16466315e-03,   1.13011456e-03,   4.30936070e-04,
            -6.38214169e-04,  -3.28756456e-04,   1.14328828e-03,
            -9.67477961e-04,   2.71635624e-05,   9.02603277e-04,
            -1.24091012e-03,   9.36695847e-04,  -3.36791788e-04,
            -1.39376472e-04,   2.48501771e-04,   9.72994947e-06,
            -4.68086179e-04,   9.15811466e-04,  -1.19611515e-03,
             1.24712034e-03,  -1.09317916e-03,   8.09882193e-04,
            -4.85460778e-04,   1.92469240e-04,   2.58831364e-05,
            -1.56809385e-04,   2.11180291e-04,  -2.13625798e-04,
             1.92838825e-04,  -1.73819747e-04,   1.72584595e-04,
            -1.93221000e-04,   2.27047318e-04,  -2.53823616e-04,
             2.45243623e-04,  -1.71057137e-04,   7.89224447e-06,
             2.50020724e-04,  -5.80505782e-04,   9.27567896e-04,
            -1.20541592e-03,   1.31733428e-03,  -1.19049652e-03,
             8.19677790e-04,  -3.01755624e-04,  -1.65905954e-04,
             3.47373889e-04,  -9.06160422e-05,  -5.35575176e-04,
             1.18572418e-03,  -1.36576866e-03,   7.61110766e-04,
             4.05397153e-04,  -1.32920480e-03,   1.16742130e-03,
             4.61076267e-05,  -1.01123275e-03,   2.40423380e-04,
             1.86428238e-03,  -2.52540353e-03,  -8.80500481e-05,
             2.80987966e-03,  -6.10059058e-04,  -3.36160557e-03,
            -6.09369052e-04,   6.98799645e-03,   2.79522471e-03,
            -9.31811920e-03,  -1.63267043e-02,  -7.09507320e-03,
             2.92192300e-02,   8.10132130e-02,   1.35136428e-01,
             1.76067004e-01,   1.86822622e-01,   1.77338136e-01,
             1.61777050e-01,   1.41570654e-01,   1.15929310e-01,
             7.97378812e-02,   3.71421913e-02])

        self.assertTrue(np.allclose(U[-1,:], correctresult))

    def test_leapfrog1d_periodic(self):
        """ test 1D CenteredDifferenceScheme1 using leapfrog time
        discretization on the wave equation with periodic boundary
        conditions """

        p = self.params
        scheme = pde.CenteredDifferenceScheme1(p['n'], p['h']) \
                    .append(pde.PeriodicBoundary1())
        A = scheme.matrix() * p['k']**2 * p['c']
        
        x = p['x']
        T = p['T']

        U = np.empty((len(T), A.shape[0]))

        U[0] = p['f'](x)
        U[1] = p['f'](x) + p['g'](x) * p['k']
        
        i = 2
        for t in T[2:]:
            U[i] = np.dot(A, U[i-1]) - U[i-2] + 2*U[i-1]
            i += 1

        correctresult = np.array([
            9.95465550e-01,   9.71323278e-01,   9.09689942e-01,
             8.19907884e-01,   7.17381252e-01,   5.96347166e-01,
             4.57793688e-01,   3.17053172e-01,   1.91615110e-01,
             1.00232280e-01,   4.18914471e-02,   6.93688769e-03,
            -5.79774181e-03,  -6.30027585e-03,  -2.36272487e-04,
             4.59826554e-03,   5.82819208e-04,  -1.92329093e-03,
            -1.21583610e-03,   1.79549424e-03,   6.99946413e-04,
            -2.16466315e-03,   1.13011456e-03,   4.30936070e-04,
            -6.38214169e-04,  -3.28756456e-04,   1.14328828e-03,
            -9.67477961e-04,   2.71635624e-05,   9.02603277e-04,
            -1.24091012e-03,   9.36695847e-04,  -3.36791788e-04,
            -1.39376472e-04,   2.48501771e-04,   9.72994947e-06,
            -4.68086179e-04,   9.15811466e-04,  -1.19611515e-03,
             1.24712034e-03,  -1.09317916e-03,   8.09882193e-04,
            -4.85460778e-04,   1.92469240e-04,   2.58831364e-05,
            -1.56809385e-04,   2.11180291e-04,  -2.13625798e-04,
             1.92838825e-04,  -1.73819747e-04,   1.72584595e-04,
            -1.93221000e-04,   2.27047318e-04,  -2.53823616e-04,
             2.45243623e-04,  -1.71057137e-04,   7.89224447e-06,
             2.50020724e-04,  -5.80505782e-04,   9.27567896e-04,
            -1.20541592e-03,   1.31733428e-03,  -1.19049652e-03,
             8.19677790e-04,  -3.01755624e-04,  -1.65905954e-04,
             3.47373889e-04,  -9.06160422e-05,  -5.35575176e-04,
             1.18572418e-03,  -1.36576866e-03,   7.61110766e-04,
             4.05397153e-04,  -1.32920480e-03,   1.16742130e-03,
             4.61076267e-05,  -1.01123275e-03,   2.40423380e-04,
             1.86428238e-03,  -2.52540353e-03,  -8.80501220e-05,
             2.80987816e-03,  -6.10080098e-04,  -3.36182494e-03,
            -6.11114485e-04,   6.97725530e-03,   2.74431128e-03,
            -9.49782725e-03,  -1.67435563e-02,  -7.32066600e-03,
             3.23634810e-02,   9.86081793e-02,   1.92875604e-01,
             3.15219623e-01,   4.50612209e-01,   5.87935658e-01,
             7.12300057e-01,   8.15440699e-01,   9.04257882e-01,
             9.68721406e-01,   9.95409070e-01])

        self.assertTrue(np.allclose(U[-1,:], correctresult))

class CenteredDifference2_FD_Tests(unittest.TestCase):

    def test_leapfrog2d(self):
        """ tests CenteredDifferenceScheme2 using a leapfrog time
        discretization to solve the wave equation with homogeneous Dirichlet
        boundary conditions. """
        n = 32
        c = 1.0
        x = np.linspace(-2*np.pi, 2*np.pi, n)
        dx = x[1] - x[0]
        dt = 0.9 * dx / np.sqrt(2)
        T = np.arange(0, 10+dt/2, dt)
        
        scheme = pde.CenteredDifferenceScheme2((n, n), (dx, dx))
        L = scheme.matrix() * c * dt**2
        
        def f(x, y, c):
            X, Y = np.meshgrid(y, x)
            return np.exp(-X**2/4 - Y**2/4)

        def g(x, y, c):
            return np.zeros((len(x), len(y)))
        
        U = np.empty((len(T), n, n))
        U[0] = f(x, x, c)
        U[1] = U[0] + g(x, x, c) * dt
        
        i = 2
        for t in T[2:]:
            u = L * U[i-1].ravel() - U[i-2].ravel() + 2*U[i-1].ravel()
            U[i] = u.reshape(U[i].shape)
            i += 1

        # reference solution near the center of the domain for U[-1, 10:21, 10:21]
        correctresult = np.array([
            [-0.24629489, -0.22554898, -0.20388323, -0.18544319, -0.17250381,
             -0.16593837, -0.16593837, -0.17250381, -0.18544319, -0.20388323,
             -0.22554898],
            [-0.22554898, -0.20600158, -0.18471238, -0.16616304, -0.15295736,
             -0.14620124, -0.14620124, -0.15295736, -0.16616304, -0.18471238,
             -0.20600158],
            [-0.20388323, -0.18471238, -0.16321971, -0.14417055, -0.13045646,
             -0.12339309, -0.12339309, -0.13045646, -0.14417055, -0.16321971,
             -0.18471238],
            [-0.18544319, -0.16616304, -0.14417055, -0.12445872, -0.11015546,
             -0.10275227, -0.10275227, -0.11015546, -0.12445872, -0.14417055,
             -0.16616304],
            [-0.17250381, -0.15295736, -0.13045646, -0.11015546, -0.09535191,
             -0.08766529, -0.08766529, -0.09535191, -0.11015546, -0.13045646,
             -0.15295736],
            [-0.16593837, -0.14620124, -0.12339309, -0.10275227, -0.08766529,
             -0.07981927, -0.07981927, -0.08766529, -0.10275227, -0.12339309,
             -0.14620124],
            [-0.16593837, -0.14620124, -0.12339309, -0.10275227, -0.08766529,
             -0.07981927, -0.07981927, -0.08766529, -0.10275227, -0.12339309,
             -0.14620124],
            [-0.17250381, -0.15295736, -0.13045646, -0.11015546, -0.09535191,
             -0.08766529, -0.08766529, -0.09535191, -0.11015546, -0.13045646,
             -0.15295736],
            [-0.18544319, -0.16616304, -0.14417055, -0.12445872, -0.11015546,
             -0.10275227, -0.10275227, -0.11015546, -0.12445872, -0.14417055,
             -0.16616304],
            [-0.20388323, -0.18471238, -0.16321971, -0.14417055, -0.13045646,
             -0.12339309, -0.12339309, -0.13045646, -0.14417055, -0.16321971,
             -0.18471238],
            [-0.22554898, -0.20600158, -0.18471238, -0.16616304, -0.15295736,
             -0.14620124, -0.14620124, -0.15295736, -0.16616304, -0.18471238,
             -0.20600158]])

        self.assertTrue(np.allclose(U[-1,10:21,10:21], correctresult))
        return



class TypeCheckTests(unittest.TestCase):

    def setUp(self):
        def f(a, b, c):
            """ <a::function, b::float[], c::int> """
            return
        self.f1 = f
        return

    def apply_verifytypes(self, f, *args):
        """ Applies `verifytypes` function to *f* and *args*. Returns
        exception if TypeError is raised. Otherwise, returns None. """
        res = None
        g = typecheck.verifytypes(f)
        try:
            g(*args)
        except TypeError as e:
            res = e
        finally:
            return res

    def test_failure1(self):
        a = lambda a: a**2
        b = np.ones(5).astype(int)
        c = 4
        
        res = self.apply_verifytypes(self.f1, a, b, c)
        self.assertIsInstance(res, TypeError)
        return

    def test_failure2(self):
        a = lambda a: a**2
        b = np.ones(5).astype(float)
        c = 4.2
        
        res = self.apply_verifytypes(self.f1, a, b, c)
        self.assertIsInstance(res, TypeError)
        return

    def test_success1(self):
        a = lambda a: a**2
        b = np.ones(5).astype(float)
        c = 4

        res = self.apply_verifytypes(self.f1, a, b, c)
        self.assertIs(res, None)
        return


if __name__ == "__main__":
    unittest.main()

