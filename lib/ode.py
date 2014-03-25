
from typecheck import verifytypes

@verifytypes
def rk4step(f, x, y, h):
    """ Perform a single step of Runge-Kutta 4 integration for either a single
    equation or a system of equations.

    *f* should be a function y' = f(x,y)
    x is the independent variable
    y is the dependent variable (the solution space)
    h is the space/time step

    <f::function, x::float[], y::float[], h::float>
    """
    k1 = h*f(x, y)
    k2 = h*f(x+0.5*h, y+0.5*k1)
    k3 = h*f(x+0.5*h, y+0.5*k2)
    k4 = h*f(x+h, y+k3)
    return (k1 + 2*k2 + 2*k3 + k4)/6.0

@verifytypes
def tvdrk3step(f, u, h):
    """ Strong stability-preserving Runge-Kutta 3 integration.

    *f* should be a function y' = f(u)
    u is the dependent variable (the solution space)
    h is the space/time step

    <f::function, u::float[], h::float>
    """
    u1 = u + h*f(u)
    u2 = 0.75*u + 0.25*(u1 + h*f(u1))
    return (u/3.0) + 2.0/3.0 * (u2 + h*f(u2))

