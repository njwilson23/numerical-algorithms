module IterativeSolvers

function solve_iter(P::AbstractArray, A::AbstractArray, x::Array, b::Array, n::Int)
    for i=1:n
        x = P \ ((P-A)*x + b)
    end
    return x
end

# Jacobi iteration
# Iteratively approximate Ax=b as
# Px = (P-A)x + b
# where P = diag(A).
# Optionally, performs weighted Jacobi, where P = diag(A)/w
function jacobi(A, x, b, n; w=1.0)
    P = spdiagm(diag(A) / w, 0)
    x = solve_iter(P, A, x, b, n)
    return x
end

weightedjacobi(A, x, b, n) = jacobi(A, x, b, n, w=2.0/3.0)

# Gauss-Seidel iteration
# Preconditioner P = diag(A) + tril(A)
function gaussseidel(A, x, b, n)
    P = tril(A)
    x = solve_iter(P, A, x, b, n)
    return x
end


# Test
n = 32
A = spdiagm((-ones(n-1), 2ones(n), -ones(n-1)), (-1, 0, 1))
b = zeros(n)
b[n/4:3n/4] = 1.0
b = sin(linspace(0, 2pi, n))
x_direct = A\b

x0 = randn(n)
x_iter = gaussseidel(A, x0, b, 200)
err = maximum(x_direct - x_iter)
@printf("error: %2.3e\n", err)

end #module




