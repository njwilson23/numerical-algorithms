
def rk4step(f, x, y, h):
    """ Perform a single step of Runge-Kutta 4 integration for either a single
    equation or a system of equations.

    <f::Function, x::Float[], y::Float[], h::Float>
    """
    k1 = h*f(x, y)
    k2 = h*f(x+0.5*h, y+0.5*k1)
    k3 = h*f(x+0.5*h, y+0.5*k2)
    k4 = h*f(x+h, y+k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)/6.0

