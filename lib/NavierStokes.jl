module NavierStokes
using DiscreteOperators
using Poisson

# vorticity formulation of Navier Stokes in 2D with periodic boundaries
#   arguments are
#       - the vorticity field
#       - the Reynolds number
#       - # grid cells
#       - grid spacing
function nvs(omega::Array, Re::Float64, nx, dx)
    psi = poisson2d(omega, nx, dx)
    omega_x = centereddiff(omega, 2) / 2dx
    omega_y = centereddiff(omega, 1) / 2dx
    psi_x = centereddiff(psi, 2) / 2dx
    psi_y = centereddiff(psi, 1) / 2dx
    return psi_y .* omega_x - psi_x .* omega_y + 1/Re .* laplacian(omega) / dx^2
end

end #module
