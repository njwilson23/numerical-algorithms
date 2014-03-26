module DiscreteOperators
export centereddiff, laplacian, gradient

# the following functions are implemented in iterative form because I find it
# easier to understand than complicated matrix operations
# assume periodic boundary conditions
function centereddiff(M::Array, axis::Integer)
    if axis == 2
        M = M'
    end

    A = zeros(typeof(M[1]), size(M))
    m,n = size(A)

    for i = 2:m-1
        for j = 1:n
            A[i,j] = M[i+1,j] - M[i-1,j]
        end
    end

    # apply boundary conditions along the first and last row
    for j = 1:n
        A[1,j] = M[2,j] - M[m,j]
        A[m,j] = M[1,j] - M[m-1,j]
    end

    if axis == 2
        A = A'
    end
    return A
end

function gradient(M::Array)
    return centereddiff(M, 1) + centereddiff(M, 2)
end

# assume periodic boundary conditions
function laplacian(M::Array)

    A = zeros(typeof(M[1]), size(M))
    m,n = size(A)

    for i = 2:m-1
        for j = 2:n-1
            A[i,j] = -4*M[i,j] + M[i+1,j] + M[i-1,j] + M[i,j+1] + M[i,j-1]
        end
    end

    # apply boundary conditions along edges
    for j = 2:n-1
        A[1,j] = -4*M[1,j] + M[2,j] + M[m,j] + M[1,j+1] + M[1,j-1]
        A[m,j] = -4*M[m,j] + M[1,j] + M[m-1,j] + M[m,j+1] + M[m,j-1]
    end

    for i = 2:m-1
        A[i,1] = -4*M[i,1] + M[i+1,1] + M[i-1,1] + M[i,2] + M[i,n]
        A[i,n] = -4*M[i,n] + M[i+1,n] + M[i-1,n] + M[i,1] + M[i,n-1]
    end

    # apply boundary conditions at corners
    A[1,1] = -4*M[1,1] + M[1,2] + M[2,1] + M[m,1] + M[1,n]
    A[1,n] = -4*M[1,n] + M[1,1] + M[2,n] + M[m,n] + M[1,n-1]
    A[m,1] = -4*M[m,1] + M[m-1,1] + M[1,1] + M[m,2] + M[m,n]
    A[m,n] = -4*M[m,n] + M[m-1,n] + M[m,n-1] + M[m,1] + M[1,n]

    return A
end

end
