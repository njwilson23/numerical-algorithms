
import unittest
import numpy as np
import sys

sys.path.append('lib')
import roots
import ode
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

class TypeCheckTests(unittest.TestCase):

    def setUp(self):
        def f(a, b, c):
            """ <a::function, b::float[], c::int> """
            return
        self.f1 = f
        return

    def apply_verifytypes(self, f, *args):
        """ Applies `verifytypes` function to *f* and *args*. Returns exception
        if TypeError is raised. Otherwise, returns None. """
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

