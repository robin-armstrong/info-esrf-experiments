using LinearAlgebra

include("../ensemble/gaspari_cohn_univariate.jl")

function fprintln(s)
    println(s)
    flush(stdout)
end

"""
    LL96_velocity!(dx, x, p, t)

Velocity field for the layered Lorenz '96 atmosphere-like system. The parameter vector has the form `p = [N_h, N_z, F_bottom, F_top, gamma]`, where
- `N_h` is the number of atmospheric columns,
- `N_z` is the number of vertical layers per column,
- `F_bottom` is the forcing strength at the bottom layer,
- `F_top` is the forcing strength at the top layer, and
- `gamma` is the coupling strength between vertical layers.

This system is from Farchi and Bocquet, "On the Efficiency of Covariance Localisation of the Ensemble Kalman Filter Using Augmented Ensembles," Frontiers in Applied Mathematics and Statistics (2019).
"""
function LL96_velocity!(dx, x, p, t)
    N_h      = round(Int64, p[1])
    N_z      = round(Int64, p[2])
    F_bottom = p[3]
    F_top    = p[4]
    gamma    = p[5]
    N_m      = N_h*N_z

    X       = reshape(x, N_h, N_z)    # each column of X represents a different vertical layer
    forcing = reshape(range(F_bottom, F_top, N_z), 1, N_z)
    forcing = repeat(forcing, outer = N_h)

    RHS = circshift(X, -1).*(circshift(X, 1) .- circshift(X, -2))
    RHS = RHS .+ gamma*[diff(X, dims = 2) zeros(N_h)] .+ gamma*[zeros(N_h) -diff(X, dims = 2)] .+ forcing
    dx[1:N_m] = vec(RHS) .- x
end

"""
    LL96_obs_operator(N_h, N_z, obs_cols, obs_centers, bandwidths)

Constructs an observation operator for the layered Lorenz '96 atmosphere-like system. Observation consists of weighted vertical integrals of atmospheric columns, similar to a satellite irradience measurement. Arguments are
- `N_h`, the number of atmospheric columns in the model.
- `N_z`, the number of vertical layers per column.
- `obs_cols`, the indices of the columns which are observed.
- `obs_centers`, the vertical indices where the weighting kernels for a single column are maximized.
- `bandwidths`, the weighting function bandwidths for a single column.

All arguments are vector-valued, and `length(bandwidths)` must equal `length(obs_centers)`.
"""
function LL96_obs_operator(N_h, N_z, obs_cols, obs_centers, bandwidths)
    ncols     = length(obs_cols)
    ncents    = length(bandwidths)
    column_op = zeros(ncents, N_z)

    for i = 1:ncents
        column_op[i, :]  = exp.(-.5*(((1:N_z) .- obs_centers[i])/bandwidths[i]).^2)
        column_op[i, :] /= norm(column_op[i, :])
    end
    
    return kron(column_op, I(N_h)[obs_cols, :])
end

"""
    LL96_stateloc(N_h, N_z, lx, lz, mu)

Constructs a state-covariance localization matrix for the layered Lorenz '96 atmosphere-like system, using the Gaspari-Cohn localization function. Arguments are
- `N_h`, the number of atmospheric columns in the model.
- `N_z`, the number of vertical layers per column.
- `lh`, the horizontal length scale for localization.
- `lz`, the vertical length scale for localization.
- `mu`, a scaling factor for combined horizontal-vertical distance.
"""
function LL96_stateloc(N_h, N_z, lh, lz, mu)
    N_m   = N_h*N_z
    L     = zeros(N_m, N_m)
    theta = range(0, 2*pi, N_h + 1)[2:end]

    wh = 1 + ceil(Int64, 2*mu*lh)
    wz = 1 + ceil(Int64, 2*mu*lz)

    for xi = 1:N_h
        for zi = 1:N_z
            for xj = (xi-wh):(xi+wh)
                for zj = (zi-wz):(zi+wz)
                    (zj < 1)   && continue
                    (zj > N_z) && continue

                    xj = mod1(xj, N_h)

                    i = Base._sub2ind((N_h, N_z), xi, zi)
                    j = Base._sub2ind((N_h, N_z), xj, zj)
                    d = sqrt((N_h*sin(.5*abs(theta[xi] - theta[xj]))/(pi*lh))^2 + ((zi - zj)/lz)^2)

                    (d <= 2*mu) && (L[i,j] = gaspari_cohn_univariate(d, mu))
                end
            end
        end
    end

    return L
end
