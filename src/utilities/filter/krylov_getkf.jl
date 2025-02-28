using LinearAlgebra
using StatsBase

include("../krylov/lanczos.jl")

"""
    krylov_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt; iternum = 10, pcrank = 20)

Ensemble adjustment Kalman filter with localization using Krylov-based matrix function evaluation. with a randomized limited-memory preconditioner for the mean update. Mandatory inputs are:
- `rng`, a random number generator for constructing the preconditioner used in the mean update.
- `mu_a` and `X_a`, preallocated arrays for the analysis mean and perturbations. These are modified in-place.
- `mu_f`, the forecast ensemble mean.
- `X_f`, normalized forecast ensemble perturbations.
- `obs`, a vector of observed data.
- `R_sqrt`, the symmetric square-root of the observation error covariance matrix.
- `H`, a matrix representing the linearized observation operator normalized by observation error.
- `BHt`, a matrix equal to `B*H'` where `B` is the background covariance in state space.
- `HBHt`, a matric equal to `H*B*H'`.

Optional inputs are:
- `iternum`, the number of iterations used to construct Krylov subspaces.
- `pcrank`, the number of Ritz vectors used in the limited-memory preconditioner.

This filter is from by Steward, Roman, Davina, and Aksoy, "Parallel Direct Solution of the Covariance-Localized Ensemble Square Root Kalman Filter Equations with Matrix Functions," Monthly Weather Review (2018).
"""
function krylov_getkf!(rng::AbstractRNG,
                       mu_a::Vector{Float64},
                       X_a::Matrix{Float64},
                       mu_f::Vector{Float64},
                       X_f::Matrix{Float64},
                       obs::Vector{Float64},
                       R_sqrt::AbstractMatrix{T1},
                       H::AbstractMatrix{T2},
                       BHt::AbstractMatrix{T3},
                       HBHt::AbstractMatrix{T4};
                       iternum::Int = 10,
                       pcrank::Int = 20) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}

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

    # normalized obs-space perturbations

    W = H*X_f

    # allocating memory for partial Krylov factorizations

    alpha = zeros(N_e, iternum)
    beta  = zeros(N_e, iternum - 1)
    V     = zeros(N_e, N_d, iternum)

    # scratch-space that we'll need later

    x1 = zeros(iternum)
    x2 = zeros(N_d)

    # Lanczos process

    prods      = HBHt*W
    lanczos    = lanczos_initialize(W, prods)
    V[:,:,1]   = lanczos.Q'
    alpha[:,1] = lanczos.alpha

    while lanczos.iteration < iternum
        # orthogonalizing the Lanczos vectors

        for i = 1:N_e
            Vk = view(V, i, :, 1:lanczos.iteration)
            xk = view(x1, 1:lanczos.iteration)
            v  = view(lanczos.testvectors, :, i)

            mul!(xk, Vk', v)
            mul!(x2, Vk, xk)
            v .-= x2
        end

        mul!(prods, HBHt, lanczos.testvectors)
        lanczos_update!(lanczos, prods)
        
        V[:,:,lanczos.iteration]      = lanczos.Q'
        alpha[:,lanczos.iteration]    = lanczos.alpha
        beta[:,lanczos.iteration - 1] = lanczos.beta
    end

    # perturbation updates

    f(x) = 1 .+ x .+ sqrt.(1 .+ x)
    U    = zeros(N_d, iternum)

    for i = 1:N_e
        eig = eigen(SymTridiagonal(alpha[i,:], beta[i,:]))
        Vi  = view(V, i, :, :)
        
        mul!(U, Vi, eig.vectors)
        broadcast!(f, eig.values, eig.values)

        w  = view(W, :, i)
        xf = view(X_f, :, i)
        xa = view(X_a, :, i)

        copy!(xa, xf)
        mul!(x1, U', w)
        lmul!(inv(Diagonal(eig.values)), x1)
        mul!(x2, U, x1)
        mul!(xa, BHt, x2, -1, 1)
    end
end
