using MKL
using DifferentialEquations
using LinearAlgebra
using CairoMakie
using StatsBase
using Random
using Printf
using TSVD
using JLD2

include("../../utilities/filter/mp_getkf.jl")
include("../../utilities/filter/svd_getkf.jl")
include("../../utilities/filter/info_esrf.jl")
include("../../utilities/filter/krylov_getkf.jl")
include("../../utilities/filter/sensrf.jl")
include("../../utilities/dynamics/LL96.jl")
include("../../utilities/ensemble/inflate.jl")
include("../../utilities/misc/setup_arrays.jl")
include("../../utilities/misc/fprints.jl")
include("../../utilities/misc/plot_config.jl")

##########################################################################################################
############################# SCRIPT PARAMETERS ##########################################################
##########################################################################################################

destination  = "src/experiments/cycled_da/cycled_da"
readme       = "Comparing B-localization methods in a cycled data assimilation experiment."

# basic parameters of the experiment

plot_only = false               # set 'true' to simply read data and plot it
rngseed   = 1                   # seed for random number generator
krange    = [4, 6, 8, 10]       # localization parameter values
numtrials = 10                  # trials per localization parameter value
pfreq     = 100                 # frequency for printing diagnostics

# parameters controlling the underlying dynamics

N_h   = 40      # horizontal dimension of grid
N_z   = 32      # vertical dimension of grid
F_b   = 8.      # forcing at bottom layer
F_t   = 4.      # forcing at top layer
gamma = 1.      # coupling between layers
T_sp  = 10.     # spinup runtime for state generation

# parameters controlling data collection

obs_cols    = 5:5:40                        # column indices that we observe
obs_centers = 6:6:30                        # channel centers within column
bandwidths  = 8*ones(length(obs_centers))   # channel bandwidths
data_noise  = 0.5               		    # standard deviation of observation error

# parameters controlling the DA cycle

N_e         = 40        # ensemble size
T_da        = 0.05      # DA cycle length
N_cycles    = 5000      # total number of cycles

# parameters controlling localization and inflation

lh     = 3      # horizontal localization length scale
lz     = 3      # vertical localization length scale
mu     = 1.     # support width for Gaspari-Cohn function
infl   = 0.01   # inflation parameter
pcrank = 10     # number of Ritz vectors for preconditioner
scale  = 300.   # scaling parameter for quadrature

########################################################################################################
################################ DATA GENERATION #######################################################
########################################################################################################

function run_cycled_experiment(rngseed, krange, numtrials, pfreq, N_h, N_z, F_b, F_t, gamma, T_sp, obs_cols,
                               obs_centers, bandwidths, data_noise, N_e, T_da, N_cycles, lh, lz, mu,
                               infl, pcrank, scale, destination, readme)
    run = Dict()

    if length(ARGS) > 0
        job_idx      = parse(Int64, ARGS[1])
        destination *= "_"*ARGS[1]

        for alg in ["sensrf", "mp_getkf", "svd_getkf", "krylov_getkf", "info_esrf"]
            run[alg] = false
        end

        if job_idx == 1
            run["sensrf"] = true
        elseif job_idx == 2
            run["mp_getkf"] = true
        elseif job_idx == 3
            run["svd_getkf"] = true
        elseif job_idx == 4
            run["krylov_getkf"] = true
        elseif job_idx == 5
            run["info_esrf"] = true
        end
    else
        for alg in ["sensrf", "mp_getkf", "svd_getkf", "krylov_getkf", "info_esrf"]
            run[alg] = true
        end
    end

    # recording parameters for this experiment

    logstr  = "rngseed     = "*string(rngseed)*"\n"
    logstr *= "krange      = "*string(krange)*"\n"
    logstr *= "numtrials   = "*string(numtrials)*"\n"
    logstr *= "N_h         = "*string(N_h)*"\n"
    logstr *= "pfreq       = "*string(pfreq)*"\n"
    logstr *= "N_z         = "*string(N_z)*"\n"
    logstr *= "F_b         = "*string(F_b)*"\n"
    logstr *= "F_t         = "*string(F_t)*"\n"
    logstr *= "gamma       = "*string(gamma)*"\n"
    logstr *= "T_sp        = "*string(T_sp)*"\n"
    logstr *= "obs_cols    = "*string(obs_cols)*"\n"
    logstr *= "obs_centers = "*string(obs_centers)*"\n"
    logstr *= "bandwidths  = "*string(bandwidths)*"\n"
    logstr *= "data_noise  = "*string(data_noise)*"\n"
    logstr *= "N_e         = "*string(N_e)*"\n"
    logstr *= "T_da        = "*string(T_da)*"\n"
    logstr *= "N_cycles    = "*string(N_cycles)*"\n"
    logstr *= "lh          = "*string(lh)*"\n"
    logstr *= "lz          = "*string(lz)*"\n"
    logstr *= "mu          = "*string(mu)*"\n"
    logstr *= "infl        = "*string(infl)*"\n"
    logstr *= "pcrank      = "*string(pcrank)*"\n"
    logstr *= "scale       = "*string(scale)*"\n"
    logstr *= "destination = "*string(destination)*"\n"
    logstr *= "\n"*readme*"\n"

    logfile = destination*"_log.txt"
    touch(logfile)
    io = open(logfile, "w")
    write(io, logstr)
    close(io)

    fprintln("\n"*logstr)

    # setting up the expermient

    rng    = MersenneTwister(rngseed)
    p      = [N_h, N_z, F_b, F_t, gamma]
    N_m    = N_h*N_z
    H_raw  = LL96_obs_operator(N_h, N_z, obs_cols, obs_centers, bandwidths)
    N_d    = length(obs_cols)*length(obs_centers)
    R_sqrt = Diagonal(data_noise*ones(N_d))

    H = R_sqrt\H_raw     # normalized observation operator

    fprintln("building localization matrices...")

    L_state = LL96_stateloc(N_h, N_z, lh, lz, mu)   # localization matrices
    L_dom   = LL96_domainloc(N_h, N_z, obs_cols, obs_centers, lh, lz, mu)

    fprintln("computing SVD of state-space localization matrix...")

    U, S, _ = tsvd(L_state, maximum(krange))

    fprintln("beginning cycled DA experiments...\n")

    # setting up memory to record experiment results

    rmse    = Dict()
    spread  = Dict()
    runtime = Dict()

    for alg in ["base", "sensrf", "mp_getkf", "svd_getkf", "krylov_getkf", "info_esrf"]
        rmse[alg]    = zeros(length(krange), numtrials, N_cycles)
        spread[alg]  = zeros(length(krange), numtrials, N_cycles)
        runtime[alg] = 0.
    end

    lmax = zeros(length(krange), numtrials, N_cycles)
    lmin = zeros(length(krange), numtrials, N_cycles)

    # preallocating memory for some vectors and matrices that we'll need to recompute many times

    mu_f = zeros(N_m)
    mu_a = zeros(N_m)
    X_f  = zeros(N_m, N_e)
    X_a  = zeros(N_m, N_e)
    B    = zeros(N_m, N_m)
    BHt  = zeros(N_m, N_d)
    HBHt = zeros(N_d, N_d)

    C = (I - ones(N_e, N_e)/N_e)/sqrt(N_e - 1)  # matrix that centers and normalizes an ensemble

    total_cycles = N_cycles*numtrials*length(krange)
    cyclecount   = 0

    for (k_idx, k) in enumerate(krange)
        L_lowrank = U[:, 1:k]*Diagonal(S[1:k].^0.5)

        for t_idx = 1:numtrials
            fprintln("------------------------------------------------------")
            fprintln("k-VALUE "*string(k_idx)*" OF "*string(length(krange)))
            fprintln("TRIAL "*string(t_idx)*" OF "*string(numtrials))
            fprintln("------------------------------------------------------")

            # generating the true state and ensembles

            fprintln("initializing ground-truth state...")
            truth = solve(ODEProblem(LL96_velocity!, randn(rng, N_m), [0., T_sp], p)).u[end]

            ens_base = zeros(N_m, N_e)

            for i = 1:N_e
                fprintln("initializing ensemble member "*string(i)*" of "*string(N_e)*"...")
                ens_base[:, i] = solve(ODEProblem(LL96_velocity!, randn(rng, N_m), [0., T_sp], p)).u[end]
            end

            run["info_esrf"]    && (ens_info   = deepcopy(ens_base))
            run["mp_getkf"]     && (ens_mp     = deepcopy(ens_base))
            run["svd_getkf"]    && (ens_svd    = deepcopy(ens_base))
            run["krylov_getkf"] && (ens_krylov = deepcopy(ens_base))
            run["sensrf"]       && (ens_sensrf = deepcopy(ens_base))

            for c_idx = 1:N_cycles
                cyclecount += 1
                pdiag       = (cyclecount % pfreq == 0)
                
                pdiag && fprintln("\n    CYCLE "*string(cyclecount)*" OF "*string(total_cycles))
                pdiag && fprintln("        ------------------------------------------")
                pdiag && fprintln("        assimilating data:\n")

                # getting data

                obs = H_raw*truth + data_noise*randn(rng, N_d)
                
                # assimilating data

                if run["sensrf"]
                    setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens_sensrf, L_state, C, H)
                    pdiag && fprint("          sensrf        ")

                    t = @elapsed sensrf!(rng, mu_a, X_a, mu_f, X_f, obs, diag(R_sqrt).^2, H, L_state)
                    inflate!(X_f, X_a, infl)
                    
                    pdiag && fprintln(string(t)*" s")
                    ens_sensrf[:,:] = mu_a*ones(1, N_e) + sqrt(N_e - 1)*X_a
                end
                
                if run["mp_getkf"]
                    setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens_mp, L_state, C, H)
                    pdiag && fprint("          mp_getkf      ")

                    t = @elapsed mp_getkf!(mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, L_lowrank)
                    inflate!(X_f, X_a, infl)
                    
                    pdiag && fprintln(string(t)*" s")
                    ens_mp[:,:] = mu_a*ones(1, N_e) + sqrt(N_e - 1)*X_a
                end

                if run["svd_getkf"]
                    setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens_svd, L_state, C, H)
                    pdiag && fprint("          svd_getkf     ")

                    t = @elapsed svd_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, B, k*N_e)
                    inflate!(X_f, X_a, infl)
                    
                    pdiag && fprintln(string(t)*" s")
                    ens_svd[:,:] = mu_a*ones(1, N_e) + sqrt(N_e - 1)*X_a
                end

                if run["krylov_getkf"]
                    setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens_krylov, L_state, C, H)
                    pdiag && fprint("          krylov_getkf  ")

                    t = @elapsed krylov_getkf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt, pcrank = pcrank, iternum = 10)
                    inflate!(X_f, X_a, infl)
                    
                    pdiag && fprintln(string(t)*" s")
                    ens_krylov[:,:] = mu_a*ones(1, N_e) + sqrt(N_e - 1)*X_a
                end

                if run["info_esrf"]
                    setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens_info, L_state, C, H)
                    pdiag && fprint("          info_esrf     ")

                    t = @elapsed info_esrf!(rng, mu_a, X_a, mu_f, X_f, obs, R_sqrt, H, BHt, HBHt, quadsize = k, scale = scale, pcrank = pcrank, iternum = 10)

                    inflate!(X_f, X_a, infl)
                    
                    pdiag && fprintln(string(t)*" s")
                    ens_info[:,:] = mu_a*ones(1, N_e) + sqrt(N_e - 1)*X_a

                    G  = svd(HBHt).S
                    l1 = G[1]
                    l2 = G[end]

                    pdiag && fprintln("        ------------------------------------------")
                    pdiag && fprintln("        obs-space covariance spectrum (info_esrf):\n")
                    pdiag && fprintln("          HBHt lmax: "*string(l1))
                    pdiag && fprintln("          HBHt lmin: "*string(l2))
                    pdiag && fprintln("        ------------------------------------------")

                    lmax[k_idx, t_idx, c_idx] = l1
                    lmin[k_idx, t_idx, c_idx] = l2
                end

                pdiag && fprintln("        total ensemble integration time:\n")

                for alg in ["sensrf", "mp_getkf", "svd_getkf", "krylov_getkf", "info_esrf"]
                    runtime[alg] = 0.
                end
                
                t_total = @elapsed begin
                    for i = 1:N_e
                        if run["sensrf"]
                            t = @elapsed begin
                                ens_sensrf[:, i] = solve(ODEProblem(LL96_velocity!, ens_sensrf[:, i], [0., T_da], p)).u[end]
                            end

                            runtime["sensrf"] = max(t, runtime["sensrf"])
                        end

                        if run["mp_getkf"]
                            t = @elapsed begin
                                ens_mp[:, i] = solve(ODEProblem(LL96_velocity!, ens_mp[:, i], [0., T_da], p)).u[end]
                            end

                            runtime["mp_getkf"] = max(t, runtime["mp_getkf"])
                        end

                        if run["svd_getkf"]
                            t = @elapsed begin
                                ens_svd[:, i] = solve(ODEProblem(LL96_velocity!, ens_svd[:, i], [0., T_da], p)).u[end]
                            end

                            runtime["svd_getkf"] = max(t, runtime["svd_getkf"])
                        end

                        if run["krylov_getkf"]
                            t = @elapsed begin
                                ens_krylov[:, i] = solve(ODEProblem(LL96_velocity!, ens_krylov[:, i], [0., T_da], p)).u[end]
                            end

                            runtime["krylov_getkf"] = max(t, runtime["krylov_getkf"])
                        end
                        
                        if run["info_esrf"]
                            t = @elapsed begin
                                ens_info[:, i] = solve(ODEProblem(LL96_velocity!, ens_info[:, i], [0., T_da], p)).u[end]
                            end

                            runtime["info_esrf"] = max(t, runtime["info_esrf"])
                        end

                        ens_base[:, i] = solve(ODEProblem(LL96_velocity!, ens_base[:, i], [0., T_da], p)).u[end]
                    end
                end
                
                pdiag && fprintln("          "*string(t_total)*" s\n")
                pdiag && fprintln("        max integration times:\n")

                pdiag && run["sensrf"]       && fprintln("          sensrf:       "*string(runtime["sensrf"])*" s")
                pdiag && run["mp_getkf"]     && fprintln("          mp_getkf:     "*string(runtime["mp_getkf"])*" s")
                pdiag && run["svd_getkf"]    && fprintln("          svd_getkf:    "*string(runtime["svd_getkf"])*" s")
                pdiag && run["krylov_getkf"] && fprintln("          krylov_getkf: "*string(runtime["krylov_getkf"])*" s")
                pdiag && run["info_esrf"]    && fprintln("          info_esrf:    "*string(runtime["info_esrf"])*" s")

                # advancing the true state

                truth = solve(ODEProblem(LL96_velocity!, truth, [0., T_da], p)).u[end]

                pdiag && fprintln("        ------------------------------------------")
                pdiag && fprintln("        forecast statistics:\n")
                pdiag && fprintln("          filter        RMSE      spread")
                pdiag && fprintln("          ------        ----      ------")
                
                # recording forecast errors

                forecast_error  = norm(truth - vec(mean(ens_base, dims = 2)))^2/N_m
                forecast_spread = mean(var(ens_base, dims = 2))
                rmse["base"][k_idx, t_idx, c_idx]   = forecast_error
                spread["base"][k_idx, t_idx, c_idx] = forecast_spread
                pdiag && @printf "          none:         %3.2e  %3.2e\n" forecast_error forecast_spread
                flush(stdout)

                if run["sensrf"]
                    forecast_error  = norm(truth - vec(mean(ens_sensrf, dims = 2)))^2/N_m
                    forecast_spread = mean(var(ens_sensrf, dims = 2))
                    rmse["sensrf"][k_idx, t_idx, c_idx]   = forecast_error
                    spread["sensrf"][k_idx, t_idx, c_idx] = forecast_spread
                    pdiag && @printf "          sensrf:       %3.2e  %3.2e\n" forecast_error forecast_spread
                    flush(stdout)
                end

                if run["mp_getkf"]
                    forecast_error  = norm(truth - vec(mean(ens_mp, dims = 2)))^2/N_m
                    forecast_spread = mean(var(ens_mp, dims = 2))
                    rmse["mp_getkf"][k_idx, t_idx, c_idx]   = forecast_error
                    spread["mp_getkf"][k_idx, t_idx, c_idx] = forecast_spread
                    pdiag && @printf "          mp_getkf:     %3.2e  %3.2e\n" forecast_error forecast_spread
                    flush(stdout)
                end

                if run["svd_getkf"]
                    forecast_error  = norm(truth - vec(mean(ens_svd, dims = 2)))^2/N_m
                    forecast_spread = mean(var(ens_svd, dims = 2))
                    rmse["svd_getkf"][k_idx, t_idx, c_idx]   = forecast_error
                    spread["svd_getkf"][k_idx, t_idx, c_idx] = forecast_spread
                    pdiag && @printf "          svd_getkf:    %3.2e  %3.2e\n" forecast_error forecast_spread
                    flush(stdout)
                end

                if run["krylov_getkf"]
                    forecast_error  = norm(truth - vec(mean(ens_krylov, dims = 2)))^2/N_m
                    forecast_spread = mean(var(ens_krylov, dims = 2))
                    rmse["krylov_getkf"][k_idx, t_idx, c_idx]   = forecast_error
                    spread["krylov_getkf"][k_idx, t_idx, c_idx] = forecast_spread
                    pdiag && @printf "          krylov_getkf: %3.2e  %3.2e\n" forecast_error forecast_spread
                    flush(stdout)
                end

                if run["info_esrf"]
                    forecast_error  = norm(truth - vec(mean(ens_info, dims = 2)))^2/N_m
                    forecast_spread = mean(var(ens_info, dims = 2))
                    rmse["info_esrf"][k_idx, t_idx, c_idx]   = forecast_error
                    spread["info_esrf"][k_idx, t_idx, c_idx] = forecast_spread
                    pdiag && @printf "          info_esrf:    %3.2e  %3.2e\n" forecast_error forecast_spread
                    flush(stdout)
                end

                @save destination*"_data.jld2" krange numtrials rmse spread lmax
            end
        end
    end
end

if !plot_only
    run_cycled_experiment(rngseed, krange, numtrials, pfreq, N_h, N_z, F_b, F_t, gamma, T_sp, obs_cols,
                          obs_centers, bandwidths, data_noise, N_e, T_da, N_cycles, lh, lz, mu,
                          infl, pcrank, scale, destination, readme)
end

########################################################################################################
################################ PLOTTING ##############################################################
########################################################################################################

@load destination*"_data.jld2" krange numtrials rmse spread lmax

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (750, 400), fonts = (; regular = regfont))

rmse_plt = Axis(fig[1,1],
                xlabel             = L"$k$",
                xticks             = krange,
                ylabel             = "Average Forecast MSE",
                yscale             = log10,
                yticks             = [1e-1, 1, 10],
                yminorgridvisible  = true,
                yminorticksvisible = true,
                yminorticks        = IntervalsBetween(10))

limits!(rmse_plt, 3, 10.5, 1e-1, 20)

record_cycles = (N_cycles - 3999):N_cycles

for alg in ["base", "sensrf", "krylov_getkf"]
    rmse_mean = mean(rmse[alg][:,:,record_cycles])
    
    hlines!(rmse_plt, [rmse_mean], linestyle = alglines[alg], color = algcolors[alg])
    scatterlines!(rmse_plt, [3.5], [rmse_mean], linestyle = alglines[alg], color = algcolors[alg], marker = algmarkers[alg], markersize = 15, label = alglabels[alg])
end

for alg in ["mp_getkf", "svd_getkf", "info_esrf"]
    rmse_means = vec(mean(rmse[alg][:,:,record_cycles], dims = [2, 3]))
    scatterlines!(rmse_plt, krange, rmse_means, color = algcolors[alg], label = alglabels[alg], marker = algmarkers[alg], markersize = 15)
end

axislegend(rmse_plt, position = :rb, orientation = :horizontal, nbanks = 3)

spread_plt = Axis(fig[1,2],
                  xlabel             = L"$k$",
                  xticks             = krange,
                  ylabel             = "Average Forecast MSE/Variance",
                  yminorgridvisible  = true,
                  yminorticksvisible = true,
                  yminorticks        = IntervalsBetween(10))

record_cycles = (N_cycles - 3999):N_cycles

for alg in ["base", "sensrf", "krylov_getkf"]
    spread_mean = mean(spread[alg][:,:,record_cycles]./rmse[alg][:,:,record_cycles])
    
    hlines!(spread_plt, [spread_mean], linestyle = alglines[alg], color = algcolors[alg])
    scatterlines!(spread_plt, [3.5], [spread_mean], linestyle = alglines[alg], color = algcolors[alg], marker = algmarkers[alg], markersize = 15, label = alglabels[alg])
end

for alg in ["mp_getkf", "svd_getkf", "info_esrf"]
    spread_means = vec(mean(spread[alg][:,:,record_cycles]./rmse[alg][:,:,record_cycles], dims = [2, 3]))
    scatterlines!(spread_plt, krange, spread_means, color = algcolors[alg], label = alglabels[alg], marker = algmarkers[alg], markersize = 15)
end

save(destination*"_plot.pdf", fig)
