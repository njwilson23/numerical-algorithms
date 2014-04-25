module IterativeSolvers

# directly implement the iteration with a matrix solve at every step
function solve_iter(P::AbstractArray, A::AbstractArray, x::Array, b::Array, n::Int)
    for i=1:n
        x = P \ ((P-A)*x + b)
    end
    return x
end

# use the pre-solved matrix M
function solve_iter(M::AbstractArray, x::Array, c::Array, n::Int)
    for i=1:n
        x = M*x + c
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
    #M = (I - P\A)
    #c = P\b
    #x = solve_iter(M, x, c, n)
    return x
end

weightedjacobi(A, x, b, n) = jacobi(A, x, b, n, w=2.0/3.0)

# Gauss-Seidel iteration
# Preconditioner P = diag(A) + tril(A)
function gaussseidel(A, x, b, n)
    P = tril(A)
    x = solve_iter(P, A, x, b, n)
    #M = (I - P\A)
    #c = P\b
    #x = solve_iter(M, x, c, n)
    return x
end

# Conjugate gradients (Krylov iteration)
# nearly a direct translation of
# http://www-math.mit.edu/cse/codes/mit18086_cg.m
function conjgrad(A, b; maxiter=null)

    maxiter = maxiter==null? size(A,1) : maxiter
    d = copy(b) # direction
    r = copy(b) # residual
    x = 0*b     # working solution
    r2 = (r' * r)[1]

    for iter = 1:maxiter

        Ad = A*d
        alpha = (r2 / (d'*Ad)[1])
        x += alpha*d
        r -= alpha*d
        r = b - A*x
        r2prev = r2
        r2 = (r'*r)[1]
        beta = r2/r2prev
        d = r + beta*d

        if norm(r) < 1e-8
            break
        end

    end

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

