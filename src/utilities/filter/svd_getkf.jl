using LinearAlgebra
using StatsBase
using Random

"""
    svd_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, B, M)

Gain-form ensemble transform Kalman filter with localization via ensemble augmentation. This filter implements augmentation using a randomized singular value decomposition. Inputs are:
- `rng`, a random number generator.
- `mu_a` and `X_a`, preallocated arrays for the analysis mean and perturbations. These are modified in-place.
- `mu_f`, the forecast ensemble mean.
- `X_f`, normalized forecase ensemble perturbations.
- `obs`, a vector of observed data.
- `R_sqrt`, the symmetric square-root of the observation error covariance matrix.
- `H`, a matrix representing the linearized observation operator.
- `B`, the background covariance matrix in state space.
- `M`, a positive integer representing the desired size of the modulated ensemble.

This filter is from Farchi and Bocquet, "On the Efficiency of Covariance Localisation of the Ensemble Kalman Filter Using Augmented Ensembles," Frontiers in Applied Mathematics and Statistics (2019).
"""
function svd_getkf!(rng::AbstractRNG,
                    mu_a::Vector{Float64},
                    X_a::Matrix{Float64},
                    mu_f::Vector{Float64},
                    X_f::Matrix{Float64},
                    obs::Vector{Float64},
                    R_sqrt::AbstractMatrix{T2},
                    H::AbstractMatrix{T1},
                    B::AbstractMatrix{T3},
                    M::Integer) where {T1 <: Real, T2 <: Real, T3 <: Real}

    POWER_ITERS = 1

    N_m, N_e = size(X_f)
    N_d      = length(obs)

    # normalized model-data mismatch

    delta     = R_sqrt\obs
    delta[:] -= H*mu_f
    
    skdim = min(ceil(Int64, 1.1*M), N_m)
    Q     = randn(rng, N_m, skdim)
    prod  = zeros(size(Q))

    for q = 1:POWER_ITERS
        mul!(prod, B, Q)
        qrobj = qr!(prod)
        Q[:,:] = Matrix{Float64}(qrobj.Q)
    end

    mul!(prod, B, Q)

    U_hat, S, _ = svd!(Q'*prod)
    S_hat       = view(S, 1:(M-1))
    broadcast!(sqrt, S_hat, S_hat)
    
    # constructing uncentered modulation perturbations

    X_mod  = zeros(N_m, M)
    X_view = view(X_mod, :, 2:M)
    U_view = view(U_hat, :, 1:(M-1))
    
    mul!(X_view, Q, U_view)
    rmul!(X_view, Diagonal(S_hat))

    # transforming the modulated perturbations to have zero mean, using a Householder reflector
    # that brings [1, 1, ... , 1] to [sqrt(M), 0, ... , 0].

    v      = -1*ones(M)
    v[1]  += sqrt(M)
    v     /= norm(v)
    Xv     = X_mod*v
    mul!(X_mod, Xv, v', -2, 1)
    
    # computing the analysis mean and perturbations

    W_f   = H*X_f
    W_mod = H*X_mod

    q       = W_mod'*((W_mod*W_mod' + I(N_d))\delta)
    mu_a[:] = mu_f
    mul!(mu_a, X_mod, q, 1, 1)
    
    _, G, Q   = svd(W_mod)
    G       .*= G
    f(x)      = 1 .+ x .+ sqrt.(1 .+ x)
    broadcast!(f, G, G)
    
    copy!(X_a, X_f)
    Y     = Q'*(W_mod'*W_f)
    lmul!(inv(Diagonal(G)), Y)
    X_a .-= X_mod*(Q*Y)
end
