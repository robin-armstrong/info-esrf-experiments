"""
    setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens, L_state, Z, H)

Populates `mu_f` and `X_f` with ensemble mean and normalized perturbtions (respectively)
corresponding to the ensemble `ens`. Populates B with the localized covariance estimate
`L_state.*(X_f*X_f')`, and `BHt`, `HBHt` with `B*H'` and `H*B*H'`, respectively, where
`H` is a linear observation operator and `L_state` is a model-space covariance localization
matrix. Requires the matrix `Z = (I - ones(N_e, N_e)/N_e)/sqrt(N_e - 1)` where `N_e` is
the number of ensemble members; this matrix centers and normalizes an ensemble.
"""
function setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens, L_state, Z, H)
    N_e     = size(X_f, 2)
    mu_f[:] = ens*ones(N_e, 1)/N_e
    
    mul!(X_f, ens, Z)
    mul!(B, X_f, X_f')
    B .*= L_state
    mul!(BHt, B, H')
    mul!(HBHt, H, BHt)
end