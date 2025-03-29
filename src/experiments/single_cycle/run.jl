using MKL
using LinearAlgebra
using CairoMakie
using StatsBase
using Random
using JLD2

include("../../utilities/ensemble/gaspari_cohn_univariate.jl")
include("../../utilities/filter/sensrf.jl")
include("../../utilities/filter/mp_getkf.jl")
include("../../utilities/filter/svd_getkf.jl")
include("../../utilities/filter/krylov_getkf.jl")
include("../../utilities/filter/info_esrf.jl")
include("../../utilities/misc/setup_arrays.jl")
include("../../utilities/misc/fprints.jl")
include("../../utilities/misc/plot_config.jl")

#############################################################################
################## SCRIPT PARAMETERS ########################################
#############################################################################

# data saving settings

plot_only   = false
destination = "src/experiments/single_cycle/single_cycle"
readme      = "Comparing B-localization methods in a single-cycle data assimilation experiment."

# random seed

rngseed = 1

# system parameters

n     = 2000    # state vector dimension
sigma = 10      # correlation length scale
eta   = 1e-4    # noise floor

# localization tuning

tune_loc = true
lrange   = 2:2:40

# observing system

n_channels = 100
bw         = 10

# data assimilation settings

M_da      = 20                      # data assimilation ensemble size
krange    = [2, 6, 10]              # modulation factors
da_trials = 100                     # trials per modulation factor
lh        = 3.                      # horizontal localization length scale
lz        = 3.                      # vertical localization length scale
mu        = 1.                      # support width for Gaspari-Cohn function
iternum   = 2                       # iteration count for Krylov methods
pcrange   = [1, 20]                 # preconditioner rank

#############################################################################
################## DATA GENERATION ##########################################
#############################################################################

function chorddist(i, j, n)
    return n*sin(pi*abs(i - j)/n)/pi
end

function run_single_da_experiment(destination, readme, rngseed, n, sigma, eta,
                                  tune_loc, lrange, n_channels, bw, M_da, krange,
                                  da_trials, lh, lz, mu, pcrange)

    logstr  = "rngseed    = "*string(rngseed)*"\n"
    logstr *= "n          = "*string(n)*"\n"
    logstr *= "sigma      = "*string(sigma)*"\n"
    logstr *= "eta        = "*string(eta)*"\n"
    logstr *= "lrange     = "*string(lrange)*"\n"
    logstr *= "n_channels = "*string(n_channels)*"\n"
    logstr *= "bw         = "*string(bw)*"\n"
    logstr *= "M_da       = "*string(M_da)*"\n"
    logstr *= "krange     = "*string(krange)*"\n"
    logstr *= "da_trials  = "*string(da_trials)*"\n"
    logstr *= "lh         = "*string(lh)*"\n"
    logstr *= "lz         = "*string(lz)*"\n"
    logstr *= "mu         = "*string(mu)*"\n"
    logstr *= "pcrange    = "*string(pcrange)*"\n"

    logfile = destination*"_log.txt"
    touch(logfile)
    io = open(logfile, "w")
    write(io, logstr)
    close(io)

    fprintln("\n"*logstr)

    rng = MersenneTwister(rngseed)
    
    fprintln("building the forecast covariance...")

    c = zeros(n)

    for i = 1:n
        d    = chorddist(i, 1, n)
        c[i] = exp(-.5*(d/sigma)^2)
    end

    c[1] += eta

    C = zeros(n,n)

    for i = 1:n
        C[i,:] = circshift(c, i-1)
    end

    csvd  = svd(C)
    csqrt = csvd.U*Diagonal(sqrt.(csvd.S))*csvd.U'

    fprintln("building forward operator...")

    # satellite-like observations that report weighted vertical line-integrals
    
    H_raw           = zeros(n_channels, n)
    spacing         = div(n, n_channels)
    channel_centers = spacing*(1:n_channels)
    
    fprintln("channel centers: "*string(channel_centers))

    for (c_idx, c) in enumerate(channel_centers)
        for i = 1:n
            H_raw[c_idx, i] = exp(-.5*(chorddist(i, c, n)/bw)^2)
        end
    end

    r = .1*mean(diag(H_raw*C*H_raw'))
    fprintln("baseline obs variance: "*string(r))

    R      = Diagonal(r*ones(n_channels))     # obs error covariance
    R_sqrt = Diagonal(sqrt(r)*ones(n_channels))
    H      = R_sqrt\H_raw

    fprintln("building the analysis covariance...")

    CHt   = C*H'
    HCHt  = H*CHt
    shcht = svd(HCHt)

    fprintln("    HCHt eigenvalues: lmin = "*string(shcht.S[end])*", lmax = "*string(shcht.S[1])*"\n")

    # finishing the calculation of the analysis covariance

    chol   = cholesky(Hermitian(I + HCHt))
    C_a    = C - CHt*(chol\CHt')
    var_a  = diag(C_a)

    # preallocating arrays that we'll need later

    mu_f = zeros(n)
    X_f  = zeros(n, M_da)
    mu_a = zeros(n)
    B    = zeros(n, n)
    X_a  = zeros(n, M_da)
    BHt  = zeros(n, n_channels)
    HBHt = zeros(n_channels, n_channels)
    Z    = (I - ones(M_da, M_da)/M_da)/sqrt(M_da - 1)

    if tune_loc
        fprintln("tuning localization radius...\n")

        LOCTRIALS = 50
        loc_errs  = zeros(length(lrange), LOCTRIALS)

        for (l_idx, l) in enumerate(lrange)
            L    = zeros(n,n)
            lrow = zeros(n)

            for i = 1:n
                d       = chorddist(i, 1, n)
                lrow[i] = exp(-.5*(d/l)^2)
            end

            for i = 1:n
                L[i,:] = circshift(lrow, i-1)
            end

            for t_idx = 1:LOCTRIALS
                ens = csqrt*randn(rng, n, M_da)
                setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens, L, Z, H)

                # DA perturbation update

                shbht = svd(HBHt)
                g     = 1 .+ shbht.S
                U     = shbht.U
                Gp    = Diagonal(g .+ sqrt.(g))
                KpH   = BHt*(U*inv(Gp)*U'*H)
                X_a  .= X_f
                mul!(X_a, KpH, X_f, -1, 1)
                v     = diag(X_a*X_a')

                loc_errs[l_idx, t_idx] = norm((var_a - v)./var_a)^2/n
            end

            fprintln("    l = "*string(l)*", err = "*string(mean(loc_errs[l_idx, :])))
        end

        meanerrs = vec(mean(loc_errs, dims = 2))
        _, l_idx = findmin(meanerrs)
        locscale = lrange[l_idx]

        L    = zeros(n,n)
        lrow = zeros(n)

        for i = 1:n
            d       = chorddist(i, 1, n)
            lrow[i] = exp(-.5*(d/locscale)^2)
        end

        for i = 1:n
            L[i,:] = circshift(lrow, i-1)
        end

        lsvd = svd(L)

        fprintln("")

        @save destination*"_locdata.jld2" locscale L lsvd
    else
        @load destination*"_locdata.jld2" locscale L lsvd
    end

    fprintln("performing DA experiments...")

    err     = Dict()
    runtime = Dict()
    
    for alg in ["sensrf", "mp_getkf", "svd_getkf", "krylov_getkf", "info_esrf"]
        err[alg]     = zeros(length(pcrange), length(krange), da_trials)
        runtime[alg] = zeros(length(pcrange), length(krange), da_trials)
    end

    lmax = zeros(length(pcrange))

    for (pc_idx, pc) in enumerate(pcrange)
        R      = Diagonal(r*ones(n_channels))    # obs error covariance
        R_sqrt = Diagonal(sqrt(r)*ones(n_channels))
        H      = R_sqrt\H_raw

        fprintln("calculating obs-space background covariance...")

        CHt   = C*H'
        HCHt  = H*CHt
        shcht = svd(HCHt)

        scale       = shcht.S[1]
        lmax[pc_idx] = scale

        fprintln("calculating exact analysis covariance...\n")

        chol   = cholesky(Hermitian(I + HCHt))
        C_a    = C - CHt*(chol\CHt')
        var_a  = diag(C_a)
        Ca_nrm = tr(C_a)

        for (k_idx, k) in enumerate(krange)
            mvecs = lsvd.U[:, 1:k]*Diagonal(lsvd.S[1:k].^0.5)

            for trial_idx = 1:da_trials
                fprintln("\n------------------------------------------------------")
                fprintln("pc-VALUE "*string(pc_idx)*" OF "*string(length(pcrange)))
                fprintln("k-VALUE  "*string(k_idx)*" OF "*string(length(krange)))
                fprintln("TRIAL    "*string(trial_idx)*" OF "*string(da_trials))
                fprintln("------------------------------------------------------\n")
                
                ens = csqrt*randn(rng, n, M_da)
                obs = H*csqrt*randn(rng, n) + sqrt(r)*randn(rng, n_channels)

                setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens, L, Z, H)

                # dry-run so that filters get precompiled

                if (k_idx == 1) && (trial_idx == 1)
                    fprintln("    dry-running sensrf...")
                    sensrf!(rng, mu_a, X_a, mu_f, X_f, obs, diag(R), H, L)

                    fprintln("    dry-running mp_getkf...")
                    mp_getkf!(mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, mvecs)

                    fprintln("    dry-running svd_getkf...")
                    svd_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, B, M_da*k)

                    fprintln("    dry-running krylov_getkf...")
                    krylov_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt, pcrank = pc)

                    fprintln("    dry-running info_esrf...")
                    info_esrf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt, quadsize = k, scale = scale, pcrank = pc)

                    fprintln("")
                end

                # profiling the algorithms

                fprint("    sensrf       ")

                t = @elapsed sensrf!(rng, mu_a, X_a, mu_f, X_f, obs, diag(R), H, L)
                runtime["sensrf"][pc_idx, k_idx, trial_idx] = t
                fprintln(" ("*string(t)*" s)")

                var_ens = diag(X_a*X_a')

                err["sensrf"][pc_idx, k_idx, trial_idx] = norm((var_a - var_ens)./var_a)^2/n

                fprint("    mp_getkf     ")

                t = @elapsed mp_getkf!(mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, mvecs)
                runtime["mp_getkf"][pc_idx, k_idx, trial_idx] = t
                fprintln(" ("*string(t)*" s)")

                var_ens = diag(X_a*X_a')

                err["mp_getkf"][pc_idx, k_idx, trial_idx] = norm((var_a - var_ens)./var_a)^2/n

                fprint("    svd_getkf    ")

                t = @elapsed svd_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, B, M_da*k)
                runtime["svd_getkf"][pc_idx, k_idx, trial_idx] = t
                fprintln(" ("*string(t)*" s)")

                var_ens = diag(X_a*X_a')

                err["svd_getkf"][pc_idx, k_idx, trial_idx] = norm((var_a - var_ens)./var_a)^2/n

                fprint("    krylov_getkf ")

                t = @elapsed krylov_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt, iternum = iternum, pcrank = pc)
                runtime["krylov_getkf"][pc_idx, k_idx, trial_idx] = t
                fprintln(" ("*string(t)*" s)")

                var_ens = diag(X_a*X_a')

                err["krylov_getkf"][pc_idx, k_idx, trial_idx] = norm((var_a - var_ens)./var_a)^2/n

                fprint("    info_esrf    ")

                t = @elapsed info_esrf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt, quadsize = k, iternum = iternum, scale = scale, pcrank = pc)
                runtime["info_esrf"][pc_idx, k_idx, trial_idx] = t
                fprintln(" ("*string(t)*" s)")

                var_ens = diag(X_a*X_a')

                err["info_esrf"][pc_idx, k_idx, trial_idx] = norm((var_a - var_ens)./var_a)^2/n

                fprintln("    --------------------------------------")
                fprintln("    ERRORS:\n")
                fprintln("    sensrf error:        "*string(err["sensrf"][pc_idx, k_idx, trial_idx]))
                fprintln("    mp_getkf error:      "*string(err["mp_getkf"][pc_idx, k_idx, trial_idx]))
                fprintln("    svd_getkf error:     "*string(err["svd_getkf"][pc_idx, k_idx, trial_idx]))
                fprintln("    krylov_getkf error:  "*string(err["krylov_getkf"][pc_idx, k_idx, trial_idx]))
                fprintln("    info_esrf error:     "*string(err["info_esrf"][pc_idx, k_idx, trial_idx]))

                @save destination*"_data.jld2" C C_a pcrange krange err runtime lmax da_trials
            end
        end
    end
end

if !plot_only
    run_single_da_experiment(destination, readme, rngseed, n, sigma, eta,
                             tune_loc, lrange, n_channels, bw, M_da, krange,
                             da_trials, lh, lz, mu, pcrange)
end

########################################################################################################
################################ PLOTTING ##############################################################
########################################################################################################

@load destination*"_data.jld2" C C_a pcrange krange err runtime lmax da_trials

CairoMakie.activate!(visible = false, type = "pdf")

fig      = Figure(size = (700, 300), fonts = (; regular = regfont))
spectrum = Axis(fig[1,1],
                xticks             = [0, 100, 200, 300, 400],
                ylabel             = "Forecast Covariance Spectrum",
                yscale             = log10,
                yticks             = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                yminorticksvisible = true,
                yminorgridvisible  = true,
                yminorticks        = IntervalsBetween(10)
               )

lines!(spectrum, 0:399, svd(C).S[1:400], color = :black)

variance = Axis(fig[1,2],
                xlabel = "State Vector Index",
                xticks = [1, 20, 40, 60],
                yticks = [0, .2, .4, .6, .8, 1, 1.2]
               )

limits!(variance, 1, 60, 0, 1.2)

var_f    = diag(C)
var_a    = diag(C_a)

lines!(variance, 1:60, var_f[1:60], linestyle = :dash, color = :black, label = "Forecast Variance")
lines!(variance, 1:60, var_a[1:60], linestyle = :solid, color = :black, label = "Analysis Variance")
axislegend(variance, position = :rc)

save(destination*"_covar_plot.pdf", fig)

fig = Figure(size = (500, 800), fonts = (; regular = regfont))
ax  = Array{Any}(undef, length(pcrange), length(krange))

for (pc_idx, pc) in enumerate(pcrange)
    for (k_idx, k) in enumerate(krange)
        ax[pc_idx, k_idx] = Axis(fig[k_idx, pc_idx],
                                title              = "p = "*string(pc)*", k = "*string(k),
                                titlefont          = regfont,
                                xlabel             = (k_idx == 3 ? "Analysis Variance Error" : ""),
                                xscale             = log10,
                                xticks             = [.1, 1., 10],
                                xminorgridvisible  = true,
                                xminorticksvisible = true,
                                xminorticks        = IntervalsBetween(10),
                                ylabel             = (pc_idx == 1 ? "Runtime (s)" : ""),
                                yscale             = log10,
                                yticks             = [.001, .01, .1, 1],
                                yminorgridvisible  = true,
                                yminorticksvisible = true,
                                yminorticks        = IntervalsBetween(10))

        limits!(ax[pc_idx, k_idx], .03, 10, 8e-4, 1)

        for alg in ["mp_getkf", "svd_getkf"]
            mean_err = mean(err[alg][:, k_idx, :])
            mean_rtm = mean(runtime[alg][:, k_idx, :])
            
            scatter!(ax[pc_idx, k_idx], [mean_err], [mean_rtm], color = algcolors[alg], marker = algmarkers[alg], markersize = 15, label = alglabels[alg])
        end

        info_err = mean(err["info_esrf"][pc_idx, k_idx, :])
        info_rtm = mean(runtime["info_esrf"][pc_idx, k_idx, :])
        scatter!(ax[pc_idx, k_idx], [info_err], [info_rtm], color = algcolors["info_esrf"], marker = algmarkers["info_esrf"], markersize = 15, label = alglabels["info_esrf"])

        sensrf_err = mean(err["sensrf"])
        sensrf_rtm = mean(runtime["sensrf"])
        scatter!(ax[pc_idx, k_idx], [sensrf_err], [sensrf_rtm], color = algcolors["sensrf"], marker = algmarkers["sensrf"], markersize = 15, label = alglabels["sensrf"])
    
        krylov_err = mean(err["krylov_getkf"])
        krylov_rtm = mean(runtime["krylov_getkf"])
        scatter!(ax[pc_idx, k_idx], [krylov_err], [krylov_rtm], color = algcolors["krylov_getkf"], marker = algmarkers["krylov_getkf"], markersize = 15, label = alglabels["krylov_getkf"])
    end
end

axislegend(ax[2,3], position = :rt, labelsize = 12)

save(destination*"_plot.pdf", fig)