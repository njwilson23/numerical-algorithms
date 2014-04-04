module Jacobi

# Jacobi iteration:
#
# Iteratively approximate Ax=b as
# Px = (P-A)x + b
# where P = diag(A)
function jacobi_iter(A::AbstractArray, x0::Array, b::Array)
    P = spdiagm(diag(A), 0)
    return P\((P-A)*xk + b)
end

n = 16
A = spdiagm((-ones(n-1), 2ones(n), -ones(n-1)), (-1, 0, 1))
b = zeros(n)
b[n/4:3n/4] = 1.0

println(A\b)

xk = ones(n)
for i=1:200
    xk = jacobi_iter(A, xk, b)
end
println(xk)

end #module




