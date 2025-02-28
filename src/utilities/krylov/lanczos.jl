using LinearAlgebra

"""
A struct containing data for Lanczos processes with a matrix `A` and several
starting vectors.. An iteration is initialized using the `lancsoz_initialize`
function, and updated using the `lanczos_update!` function.
"""
mutable struct LanczosData
    alpha::Vector{Float64}
    beta::Vector{Float64}
    Q::Matrix{Float64}
    U::Matrix{Float64}
    R::Matrix{Float64}
    iteration::Int64
    testvectors::Matrix{Float64}
end

"""
    lanczos_initialize(B, AB)

Initializes Lanczos iterations for a matrix `A` and vectors `B[:,j]`. Inputs
are starting vectors `B` and `AB = A*B`. Returns a `LanczosData` object, which
can be passed to `lanczos_update!` to continue the iteration.
"""
function lanczos_initialize(B, AB)
    beta  = norm.(eachcol(B))
    Dbeta = Diagonal(beta)
    U     = zeros(size(B))
    Q     = B/Dbeta
    alpha = diag(Q'*AB)./beta

    R = AB/Dbeta
    mul!(R, Q, Diagonal(alpha), -1, 1)

    testvectors = R/Diagonal(norm.(eachcol(R)))
    iteration   = 1

    return LanczosData(alpha, beta, Q, U, R, iteration, testvectors)
end

"""
    lanczos_update!(lanczos, prods)

Updates Lanczos iterations for a matrix `A`. Inputs are `lanczos`, a
`LanczosData` object, and `prods = A*lanczos.testvectors`. An iteration
is initialized with `lanczos_initialize`.
"""
function lanczos_update!(lanczos::LanczosData, prods)    
    lanczos.U[:,:]   = lanczos.Q
    lanczos.Q[:,:]   = lanczos.testvectors
    lanczos.alpha[:] = diag(lanczos.Q'*prods)
    lanczos.beta[:]  = norm.(eachcol(lanczos.R))
    
    lanczos.R[:,:] = prods
    mul!(lanczos.R, lanczos.Q, Diagonal(lanczos.alpha), -1, 1)
    mul!(lanczos.R, lanczos.U, Diagonal(lanczos.beta), -1, 1)
    
    # computing the next vector for which a matrix-vector product will be requested

    lanczos.testvectors[:,:] = lanczos.R
    rdiv!(lanczos.testvectors, Diagonal(norm.(eachcol(lanczos.R))))

    lanczos.iteration += 1
end
