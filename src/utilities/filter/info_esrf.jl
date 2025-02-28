using LinearAlgebra
using StatsBase
using Random

include("../krylov/pcg.jl")
include("../krylov/lmp.jl")
include("../quadrature/quadrature_rules.jl")

"""
    info_esrf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt; scale = 1., quadsize = 5, iternum = 10, pcrank = 20)

Integral-form ensemble square-root filter with model-space localization via quadrature. A limited-memory preconditioner
based on randomized Ritz pairs is used to accelerate the linear solve at each quadrature point, and for the mean update. Mandatory
inputs are:
- `rng`, a random number generator.
- `mu_a` and `X_a`, preallocated arrays for the analysis mean and perturbations. These are modified in-place.
- `mu_f`, the forecast ensemble mean.
- `X_f`, normalized forecast ensemble perturbations.
- `obs`, a vector of observed data.
- `R_sqrt`, the symmetric square-root of the observation error covariance matrix.
- `H`, a matrix representing the linearized observation operator normalized by observation error. Dimensions should be `(N_d, N_m)`, where `N_d` is the length of the observation vector.
- `BHt`, a matrix equal to `B*H'` where `B` is the background covariance in state space.
- `HBHt`, a matric equal to `H*B*H'`.

Optional inputs are:
- `scale`, a positive real number controlling the generation of quadrature weights and nodes.
- `quadsize`, a positive integer indicating the number of quadrature points for computing the ensemble transformation.
- `iternum`, the number of iterations used to construct Krylov subspaces for linear system solves.
- `pcrank`, the number of Ritz vectors that define the preconditioner.
"""
function info_esrf!(rng::AbstractRNG,
                    mu_a::Vector{Float64},
                    X_a::Matrix{Float64},
                    mu_f::Vector{Float64},
                    X_f::Matrix{Float64},
                    obs::Vector{Float64},
                    R_sqrt::AbstractMatrix{R1},
                    H::AbstractMatrix{R2},
                    BHt::AbstractMatrix{R3},
                    HBHt::AbstractMatrix{R4};
                    scale::Real = 1.,
                    quadsize::Int = 5,
                    iternum::Int = 10,
                    pcrank::Int = 20) where {R1 <: Real, R2 <: Real, R3 <: Real, R4 <: Real}

    N_m, N_e = size(X_f)
    N_d      = length(obs)

    # normalized model-data mismatch

    res = reshape(R_sqrt\obs, N_d, 1)
    mul!(res, H, mu_f, -1, 1)

    # computing increments in obs space

    W = H*X_f

    # forming Ritz pairs for the preconditioner

    skdim = min(N_d, ceil(Int, 1.2*pcrank))
    Q     = randn(rng, N_d, skdim)
    prod  = zeros(size(Q))

    POWERS = 3

    for q = 1:POWERS
        mul!(prod, HBHt, Q)
        qrobj  = qr!(prod)
        Q[:,:] = Matrix{Float64}(qrobj.Q)
    end

    mul!(prod, HBHt, Q)
    U, S, _  = svd!(Q'*prod)
    d        = S[1:pcrank]
    Q        = Q*U[:,1:pcrank]

    # setting up the preconditioner

    dmin = minimum(diag(HBHt))
    rho  = 1 + dmin
    V    = Q*inv(Diagonal(sqrt.(1 .+ d)))
    AV   = HBHt*V
    AV .+= V
    G    = AV'*AV
    t1   = zeros(pcrank, 1)
    t2   = zeros(pcrank, 1)

    # computing the analysis mean

    x0    = zeros(N_d, 1)
    p_res = zeros(N_d, 1)
    apply_lmp!(p_res, res, rho, V, AV, G, t1, t2)
    
    cg_matvec   = HBHt*p_res
    cg_matvec .+= p_res
    cg          = pcg_initialize(x0, res, p_res, cg_matvec)

    while cg.iteration < iternum
        apply_lmp!(cg_matvec, cg.testvectors, rho, V, AV, G, t1, t2)
        pcg_update_1!(cg, cg_matvec)

        mul!(cg_matvec, HBHt, cg.testvectors)
        cg_matvec .+= cg.testvectors
        pcg_update_2!(cg, cg_matvec)
    end

    mu_a[:] = mu_f
    mul!(mu_a, BHt, vec(cg.sol), 1, 1)

    # computing weights and nodes for quadrature

    nodes, weights = setup_elliptic_quad(scale, quadsize)
    infvals        = nodes .+ 1
    sh_old         = 1.

    # setting up data structures for perturbation updates

    Z     = zeros(N_d, N_e)
    X0    = zeros(N_d, N_e)
    R0    = zeros(N_d, N_e)
    P0    = zeros(N_d, N_e)
    AP    = zeros(N_d, N_e)
    prods = zeros(N_d, N_e)
    T1    = zeros(pcrank, N_e)
    T2    = zeros(pcrank, N_e)
    cgp   = pcg_initialize(X0, R0, P0, AP)

    # perturbation updates

    for q = 1:quadsize
        # setting up the preconditioner
        
        rho = infvals[q] + dmin 

        V .= Q
        rmul!(V, inv(Diagonal(sqrt.(infvals[q] .+ d))))
        
        AV .= infvals[q]*V
        mul!(AV, HBHt, V, 1, 1)

        mul!(G, AV', AV)

        # updating the perturbations

        copy!(R0, W)
        apply_lmp!(P0, R0, rho, V, AV, G, T1, T2)
        
        AP .= infvals[q]*P0
        mul!(AP, HBHt, P0, 1, 1)
        pcg_initialize!(cgp, X0, R0, P0, AP)

        while cgp.iteration < iternum
            apply_lmp!(prods, cgp.testvectors, rho, V, AV, G, T1, T2)
            pcg_update_1!(cgp, prods)

            prods .= infvals[q]*cgp.testvectors
            mul!(prods, HBHt, cgp.testvectors, 1, 1)
            pcg_update_2!(cgp, prods)
        end

        Z .+= weights[q]*cgp.sol
    end

    X_a[:,:] = X_f
    mul!(X_a, BHt, Z, -1, 1)
end
