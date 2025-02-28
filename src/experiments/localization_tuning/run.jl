using MKL
using DifferentialEquations
using LinearAlgebra
using CairoMakie
using StatsBase
using Random
using Printf
using JLD2

include("../../utilities/dynamics/LL96.jl")
include("../../utilities/ensemble/inflate.jl")
include("../../utilities/misc/setup_arrays.jl")
include("../../utilities/misc/fprints.jl")
include("../../utilities/misc/plot_config.jl")

##########################################################################################################
############################# SCRIPT PARAMETERS ##########################################################
##########################################################################################################

destination  = "src/experiments/localization_tuning/localization_tuning"
readme       = "Comparing B-localization methods in a cycled data assimilation experiment."

# basic parameters of the experiment

plot_only = false       # set 'true' to simply read data and plot it
rngseed   = 1           # seed for random number generator
lrange    = 1:8         # localization parameter values
mu        = 1.          # support width for Gaspari-Cohn function
numtrials = 10          # trials per localization parameter value
pfreq     = 100         # frequency for printing diagnostics

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
N_cycles    = 5000      # total number of cycles per localization

# parameters controlling localization and inflation

lh     = 5.     # horizontal localization length scale
lz     = 4.     # vertical localization length scale
infl   = 0.01   # inflation parameter
scale  = 300.   # scaling parameter for quadrature

########################################################################################################
################################ DATA GENERATION #######################################################
########################################################################################################

function run_localization_tuning(rngseed, lrange, numtrials, pfreq, N_h, N_z, F_b, F_t, gamma, T_sp, obs_cols,
                                 obs_centers, bandwidths, data_noise, N_e, T_da, N_cycles, lh, lz, mu,
                                 infl, scale, destination, readme)

    # recording parameters for this experiment

    if length(ARGS) > 0
        job_idx      = parse(Int64, ARGS[1])
        destination *= "_"*ARGS[1]
        lrange       = [lrange[job_idx]]
    end

    logstr  = "rngseed     = "*string(rngseed)*"\n"
    logstr *= "lrange      = "*string(lrange)*"\n"
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

    fprintln("beginning localization tuning...\n")

    # setting up memory to record experiment results

    rmse    = zeros(length(lrange), numtrials, N_cycles)
    spread  = Dict()
    runtime = Dict()

    # preallocating memory for some vectors and matrices that we'll need to recompute many times

    mu_f = zeros(N_m)
    mu_a = zeros(N_m)
    X_f  = zeros(N_m, N_e)
    X_a  = zeros(N_m, N_e)
    B    = zeros(N_m, N_m)
    BHt  = zeros(N_m, N_d)
    HBHt = zeros(N_d, N_d)

    C = (I - ones(N_e, N_e)/N_e)/sqrt(N_e - 1)  # matrix that centers and normalizes an ensemble

    total_cycles = N_cycles*numtrials*length(lrange)
    cyclecount   = 0

    for (l_idx, l) in enumerate(lrange)
        fprintln("\nbuilding localization matrix...")
        L_state = LL96_stateloc(N_h, N_z, l, l, mu)   # localization matrix

        for t_idx = 1:numtrials
            fprintln("------------------------------------------------------")
            fprintln("l-VALUE "*string(l_idx)*" OF "*string(length(lrange)))
            fprintln("TRIAL "*string(t_idx)*" OF "*string(numtrials))
            fprintln("------------------------------------------------------")

            # generating the true state and ensembles

            fprintln("initializing ground-truth state...")
            truth = solve(ODEProblem(LL96_velocity!, randn(rng, N_m), [0., T_sp], p)).u[end]

            ens = zeros(N_m, N_e)

            for i = 1:N_e
                fprintln("initializing ensemble member "*string(i)*" of "*string(N_e)*"...")
                ens[:, i] = solve(ODEProblem(LL96_velocity!, randn(rng, N_m), [0., T_sp], p)).u[end]
            end

            for c_idx = 1:N_cycles
                cyclecount += 1
                pdiag       = (cyclecount % pfreq == 0)
                
                pdiag && fprintln("\n    CYCLE "*string(cyclecount)*" OF "*string(total_cycles))
                pdiag && fprintln("        ------------------------------------------")
                pdiag && fprint("        assimilating data: ")

                # getting data

                obs = H_raw*truth + data_noise*randn(rng, N_d)
                obs = R_sqrt\obs
                
                # assimilating data

                setup_arrays!(mu_f, X_f, B, BHt, HBHt, ens, L_state, C, H)
                
                t = @elapsed begin
                    shbht = svd(HBHt)
                    g     = 1 .+ shbht.S
                    U     = shbht.U
                    G     = Diagonal(g)
                    Gp    = Diagonal(g .+ sqrt.(g))
                    K     = BHt*(U*inv(G)*U')
                    KpH   = BHt*(U*inv(Gp)*U'*H)

                    mu_a .= mu_f + K*(obs - H*mu_f)
                    X_a  .= X_f
                    mul!(X_a, KpH, X_f, -1, 1)
                    inflate!(X_f, X_a, infl)
                    ens .= mu_a*ones(1, N_e) + sqrt(N_e - 1)*X_a
                end

                pdiag && fprintln(string(t)*" s")
                pdiag && fprintln("        ------------------------------------------")
                pdiag && fprint("        integrating ensemble: ")

                t_total = @elapsed begin
                    for i = 1:N_e
                        ens[:, i] = solve(ODEProblem(LL96_velocity!, ens[:, i], [0., T_da], p)).u[end]
                    end
                end
                
                pdiag && fprintln(string(t_total)*" s")
                
                # advancing the true state

                truth = solve(ODEProblem(LL96_velocity!, truth, [0., T_da], p)).u[end]

                pdiag && fprintln("        ------------------------------------------")
                pdiag && fprintln("        forecast statistics:\n")
                pdiag && fprintln("          RMSE      spread")
                pdiag && fprintln("          ----      ------")
                
                # recording forecast errors

                forecast_error            = norm(truth - vec(mean(ens, dims = 2)))^2/N_m
                forecast_spread           = mean(var(ens, dims = 2))
                rmse[l_idx, t_idx, c_idx] = forecast_error
                
                pdiag && @printf "          %3.2e  %3.2e\n" forecast_error forecast_spread
                flush(stdout)

                @save destination*"_data.jld2" lrange numtrials N_cycles rmse
            end
        end
    end
end

if !plot_only
    run_localization_tuning(rngseed, lrange, numtrials, pfreq, N_h, N_z, F_b, F_t, gamma, T_sp, obs_cols,
                            obs_centers, bandwidths, data_noise, N_e, T_da, N_cycles, lh, lz, mu,
                            infl, scale, destination, readme)
end

########################################################################################################
################################ PLOTTING ##############################################################
########################################################################################################

@load destination*"_data.jld2" lrange numtrials N_cycles rmse
