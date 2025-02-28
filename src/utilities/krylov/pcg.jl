using LinearAlgebra

"""
A struct containing data for a preconditioned conjugate gradients iteration with one or several right-hand sides. An iteration is initialized using the `pcg_initialize` function, and updated using the `pcg_update_1!` and `pcg_update_2!` functions. In addition to internal data needed by the iteration, `PCGData` structs contain four fields of interest to the user:
- `iteration`, the number of iterations that have been run so far.
- `testvector`, the next vector for which a matrix-vector product must be computed before the iteration can proceed.
- `sol`, a matrix of current solutions. Each column represents the current solution for one of the shifted linear systems.
- `r`, the residual vector for the most ill-conditioned system currently being solved.
"""
mutable struct PCGData
    alpha::Vector{Float64}
    beta::Vector{Float64}
    Q::Matrix{Float64}
    C::Matrix{Float64}
    iteration::Int64
    testvectors::Matrix{Float64}
    sol::Matrix{Float64}
    R::Matrix{Float64}
end

"""
    pcg_initialize(X0, R0, P0, AP)

Initializes a preconditioned conjugate gradients iteration to solve the linear systems `A*X = B` with a preconditioner `P`. Inputs are:
- Initial guesses `X0`.
- `R0 = B - A*X0`.
- `P0 = inv(P)*R0`.
- `AP  = A*P0`.
Returns a `PCGData` object. See `pcg_initialize!` for a memory-efficient version.
"""
function pcg_initialize(X0, R0, P0, AP)
    Q     = deepcopy(P0)
    C     = deepcopy(Q)
    beta  = diag(R0'*Q)
    alpha = beta./diag(C'*AP)
    sol   = C*Diagonal(alpha)
    sol .+= X0
    R     = AP*Diagonal(-alpha)
    R   .+= R0

    iteration   = 1
    testvectors = deepcopy(R)

    return PCGData(alpha, beta, Q, C, iteration, testvectors, sol, R)
end

"""
    pcg_initialize!(cg_data, X0, R0, P0, AP)

Same as `pcg_initialize`, but avoids memory allocation by overwriting `cg_data`,
a variable of type `LanczosData`.
"""
function pcg_initialize!(cg_data::PCGData, X0, R0, P0, AP)
    copy!(cg_data.Q, P0)
    copy!(cg_data.C, cg_data.Q)

    cg_data.beta  = diag(R0'*cg_data.Q)
    cg_data.alpha = cg_data.beta./diag(cg_data.C'*AP)

    copy!(cg_data.sol, X0)
    mul!(cg_data.sol, cg_data.C, Diagonal(cg_data.alpha), 1, 1)

    copy!(cg_data.R, AP)
    rmul!(cg_data.R, Diagonal(-cg_data.alpha))
    cg_data.R .+= R0

    cg_data.iteration  = 1
    copy!(cg_data.testvectors, cg_data.R)
end

"""
    pcg_update_1!(cg_data, prods)

Performs the first step of a preconditioned conjugate gradients update to solve
`A*X = B` with a preconditioner `P`. Inputs are:
- `cg_data`, an object of type `PCGData` which is modified directly by the function, and passed to the next iteration. This object can be initialized using the `pcg_initialize` function.
- `prods`, a matrix equal to `inv(P)*cg_data.testvectors`.
"""
function pcg_update_1!(cg_data::PCGData, prods)
    cg_data.beta  = diag(cg_data.R'*prods)./cg_data.beta
    rmul!(cg_data.C, Diagonal(cg_data.beta))
    cg_data.C  .+= prods
    
    copy!(cg_data.Q, prods)
    copyto!(cg_data.testvectors, cg_data.C)
end

"""
    pcg_update_1!(cg_data, prod)

Performs the second step of a preconditioned conjugate gradients update to solve
`A*X = B` with a preconditioner `P`. Inputs are:
- `cg_data`, an object of type `PCGData` which is modified directly by the function, and passed to the next iteration. This object can be initialized using the `pcg_initialize` function.
- `prods`, a matrix equal to `A*cg_data.testvectors`.
"""
function pcg_update_2!(cg_data::PCGData, prods)
    cg_data.beta    = diag(cg_data.R'*cg_data.Q)
    cg_data.alpha   = cg_data.beta./diag(cg_data.C'*prods)
    mul!(cg_data.sol, cg_data.C, Diagonal(cg_data.alpha), 1, 1)
    mul!(cg_data.R, prods, Diagonal(cg_data.alpha), -1, 1)

    copy!(cg_data.testvectors, cg_data.R)
    cg_data.iteration += 1
end
