module Poisson
export poisson1d, poisson2d

# functions to solve the problem \nabla^2 u = f in 1 or 2D using a fast
# spectral method
#   - how to use DST directly?

function poisson1d(f::Array, nx, dx)
    fhat = fft(f)
    eigenval = (2 .- 2*cos(dx*[0:nx-1])) / dx^2
    eigenval[1] = 1.0
    uhat = fhat ./ eigenval
    real(ifft(uhat))
end

function poisson2d(f::Array, nx, dx)
    fhat = fft(f)
    e1 = (2 .- 2*cos(dx*[0:nx-1])) / dx^2
    eigenval = e1 .+ e1'
    uhat = fhat ./ eigenval
    uhat[1] = 0.0   # pin the solution
    real(ifft(uhat))
end

function _applyover(L, R, nx::Int, f::Function)
    x = linspace(L, R, nx+1)[2:end]
    f(x)
end

end #module
