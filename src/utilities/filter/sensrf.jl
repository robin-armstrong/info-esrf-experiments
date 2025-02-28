using Random

"""
    sensrf!(mu_a, X_a, mu_f, X_f, obs, H, L)

Serial ensemble square-root filter with model-space localization. Inputs are:
- `mu_a` and `X_a`, preallocated arrays for the analysis mean and perturbations. These are modified in-place.
- `mu_f`, the forecast ensemble mean.
- `X_f`, normalized forecase ensemble perturbations.
- `obs`, a vector representing observed data.
- `rcoeffs`, a vector of observation error variances.
- `R_sqrt`, the symmetric square-root of the observation error covariance matrix.
- `H`, a matrix representing the linearized observation operator normalized by observation error.
- `L`, the state-space covariance localization matrix.

This filter is from Shlyaeva, Whitaker, and Snyder, "Model-Space Localization in Serial Ensemble Filters," Journal of Advances in Modeling Earth Systems (2019).
"""
function sensrf!(rng::AbstractRNG,
                 mu_a::Vector{Float64},
                 X_a::Matrix{Float64},
                 mu_f::Vector{Float64},
                 X_f::Matrix{Float64},
                 obs::Vector{Float64},
                 rcoeffs::Vector{Float64},
                 H::AbstractMatrix{R1},
                 L::AbstractMatrix{R2}) where {R1 <: Real, R2 <: Real}

    N_m, N_e = size(X_f)    # state dimension and ensemble size
    N_d      = length(obs)  # data dimension

    copy!(mu_a, mu_f)
    copy!(X_a, X_f)

    # preallocating space that we'll need later

    T1 = zeros(N_m, N_e)
    T2 = zeros(N_m, N_e)
    v  = zeros(N_m)

    # data are assimilated in random order to account for
    # noncommutativity introduced by the localization.
    
    obs_idx = randperm(rng, N_d)

    for d in obs_idx
        h = view(H, d, :)   # observation functional for the i^th data point
        w = h'*X_a          # prior obs-space perturbations for this data point

        # computing the normalized model-data mismatch

        delta = obs[d]/sqrt(rcoeffs[d]) - h'*mu_a

        # forming the localized cross-covariance vector between
        # the state and the d^th data point.

        T1 .= Diagonal(h)*X_a
        T2 .= L*T1
        T1 .= X_a.*T2
        v  .= sum(T1, dims = 2)

        # other values that we'll need

        ovar = h'*v                 # prior localized variance for this data point
        rho1 = 1 + ovar             # Kalman gain coefficient for mean update
        rho2 = rho1 + sqrt(rho1)    # Kalman gain coefficient for perturbation update

        # updating the ensemble

        mul!(mu_a, v, delta/rho1, 1, 1)     # mean update
        mul!(X_a, v, w, -1/rho2, 1)         # perturbation update
    end
end
