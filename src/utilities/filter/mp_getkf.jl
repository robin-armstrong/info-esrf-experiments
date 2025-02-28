using LinearAlgebra
using StatsBase

"""
    mp_getkf!(mu_f, X_f, obs, R_sqrt, H, mvecs)

Gain-form ensemble transform Kalman filter with localization via ensemble augmentation. Unlike `svd_getkf`, which implements augmentation using a singular value decomposition, this filter uses a modulation product. See Bishop, Whitaker, and Lei, 2017. Inputs are:
- `mu_a` and `X_a`, preallocated arrays for the analysis mean and perturbations. These are modified in-place.
- `mu_f`, the forecast ensemble mean.
- `X_f`, normalized forecast ensemble perturbations.
- `obs`, a vector of representing observed data.
- `R_sqrt`, the symmetric square-root of the observation error covariance matrix.
- `H`, a matrix representing the linearized observation operator normalized by observation error.
- `mvecs`, modulation vectors derived from a spectral decomposition of the localization matrix.

This filter is from Bishop, Whitaker, and Lei, "Gain Form of the Ensemble Transform Kalman Filter and Its Relevance to Satellite Data Assimilation with Model Space Ensemble Covariance Localization," Monthly Weather Review (2017).
"""
function mp_getkf!(mu_a::Vector{Float64},
                   X_a::Matrix{Float64},
                   mu_f::Vector{Float64},
                   X_f::Matrix{Float64},
                   obs::Vector{Float64},
                   R_sqrt::AbstractMatrix{T1},
                   H::AbstractMatrix{T2},
                   mvecs::AbstractMatrix{T3}) where {T1 <: Real, T2 <: Real, T3 <: Real}

    N_m, N_e = size(X_f)
    N_d      = length(obs)
    k        = size(mvecs, 2)

    # normalized model-data mismatch

    delta     = R_sqrt\obs
    delta[:] -= H*mu_f

    # modulation
    
    X_mod = zeros(N_m, N_e*k)
    xf    = zeros(N_m)
    
    for i = 1:N_e
        xf     = view(X_f, :, i)
        X_view = view(X_mod, :, (1:k) .+ (i - 1)*k)
        
        copy!(X_view, mvecs)
        lmul!(Diagonal(xf), X_view)
    end

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
