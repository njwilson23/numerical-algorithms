module tests

using Base.Test
import RungeKutta: rk4step, tvdrk3step

# Approximates `e`
function test_rk4()
    x = 1.0
    f(x) = x
    for i=1:1000
        x += rk4step(f, x, 0.001)
    end
    return x
end
@test_approx_eq(test_rk4(), e)

function test_rk3()
    x = 1.0
    f(x) = x
    for i=1:5000
        x += tvdrk3step(f, x, 0.0002)
    end
    return x
end
@test_approx_eq(test_rk3(), e)

end #module
