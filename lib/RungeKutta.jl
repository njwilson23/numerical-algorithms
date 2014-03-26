module RungeKutta
export tvdrk3step, rk4step

# "Strong stability preserving RK3"
function tvdrk3step(f::Function, u::Union(FloatingPoint, Array), h::FloatingPoint)
    u1 = u + h*f(u)
    u2 = 0.75u + 0.25 * (u1 + h*f(u1))
    return u/3.0 + 2.0/3.0 * (u2 + h*f(u2)) - u
end

function rk4step(f::Function, u::Union(FloatingPoint, Array), h::FloatingPoint)
    k1 = h*f(u)
    k2 = h*f(u + 0.5k1)
    k3 = h*f(u + 0.5k2)
    k4 = h*f(u + k3)
    return (k1 + 2k2 + 2k3 + k4) / 6.0
end

end #module
