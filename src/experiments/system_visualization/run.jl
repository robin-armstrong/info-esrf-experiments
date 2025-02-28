using MKL
using DifferentialEquations
using LinearAlgebra
using CairoMakie
using Random
using JLD2

include("../../utilities/dynamics/LL96.jl")
include("../../utilities/misc/fprints.jl")
include("../../utilities/misc/plot_config.jl")

#############################################################################
################## SCRIPT PARAMETERS ########################################
#############################################################################

destination = "src/experiments/system_visualization/system"
plot_only   = false
readme      = "Generating a Hovmoeller diagram for the layered Lorenz'96 system."

rngseed  = 1
N_h      = 40       # horizontal dimension of grid
N_z      = 32       # vertical dimension of grid
F_b      = 8.       # forcing at bottom layer
F_t      = 4.       # forcing at top layer
gamma    = 1.       # coupling between layers
T_spinup = 50.
T_record = 20.
N_snaps  = 200

obs_centers = 6:6:30                        # channel centers within column
bandwidths  = 8*ones(length(obs_centers))   # channel bandwidths

#############################################################################
################## DATA GENERATION ##########################################
#############################################################################

function run_state_data(destination, readme, rngseed, N_h, N_z, F_b, F_t,
                        gamma, T_spinup, T_record, N_snaps, obs_centers,
                        bandwidths)
    logstr  = "rngseed     = "*string(rngseed)*"\n"
    logstr *= "N_h         = "*string(N_h)*"\n"
    logstr *= "N_z         = "*string(N_z)*"\n"
    logstr *= "F_b         = "*string(F_b)*"\n"
    logstr *= "F_t         = "*string(F_t)*"\n"
    logstr *= "gamma       = "*string(gamma)*"\n"
    logstr *= "T_spinup    = "*string(T_spinup)*"\n"
    logstr *= "T_record    = "*string(T_record)*"\n"
    logstr *= "N_snaps     = "*string(N_snaps)*"\n"
    logstr *= "obs_centers = "*string(obs_centers)*"\n"
    logstr *= "bandwidths  = "*string(bandwidths)*"\n"
    logstr *= "\n"*readme*"\n"

    logfile = destination*"_log.txt"
    touch(logfile)
    io = open(logfile, "w")
    write(io, logstr)
    close(io)

    fprintln("\n"*logstr)

    rng = MersenneTwister(rngseed)
    N_m = N_h*N_z
    p   = [N_h, N_z, F_b, F_t, gamma]
    
    fprintln("spinning up the initial state...")
    x = solve(ODEProblem(LL96_velocity!, randn(rng, N_m), [0., T_spinup], p)).u[end]

    fprintln("recording time series...")

    snaps = zeros(N_z, N_snaps)

    for t = 1:N_snaps
        fprintln("    snapshot "*string(t)*" of "*string(N_snaps)*"...")

        X          = reshape(x, N_h, N_z)
        snaps[:,t] = X[1,:]
        x[:]       = solve(ODEProblem(LL96_velocity!, x, [0., T_record/N_snaps], p)).u[end]
    end

    num_channels = length(obs_centers)
    weights      = zeros(num_channels, N_z)

    for i = 1:num_channels
        weights[i, :]  = exp.(-.5*(((1:N_z) .- obs_centers[i])/bandwidths[i]).^2)
        weights[i, :] /= norm(weights[i, :])
    end

    @save destination*"_data.jld2" snaps x weights obs_centers N_h N_z
end

if !plot_only
    run_state_data(destination, readme, rngseed, N_h, N_z, F_b, F_t,
                   gamma, T_spinup, T_record, N_snaps, obs_centers,
                   bandwidths)
end

#############################################################################
################## PLOTTING #################################################
#############################################################################

@load destination*"_data.jld2" snaps x weights obs_centers N_h N_z

smin = minimum(snaps)
smax = maximum(snaps)
xmin = minimum(x)
xmax = maximum(x)

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (500, 800), fonts = (; regular = regfont))

state = Axis(fig[1,1],
             xlabel = "Horizontal Index",
             xticks = [1, 10, 20, 30, 40],
             ylabel = "Vertical Index",
             yticks = [1, 8, 16, 24, 32])

heatmap!(state, 1:40, 1:32, reshape(x, N_h, N_z), colorrange = (xmin, xmax), rasterize = true)
Colorbar(fig[1,2], colorrange = (smin, smax))

hov = Axis(fig[2,1],
           xlabel = "Model Time",
           ylabel = "Vertical Index",
           yticks = [1, 8, 16, 24, 32])

heatmap!(hov, range(0, T_record, N_snaps), 1:32, transpose(snaps), colorrange = (smin, smax), rasterize = true)
Colorbar(fig[2,2], colorrange = (smin, smax))

fwd = Axis(fig[3,1],
           xlabel = L"Vertical Index ($j$)",
           xticks = [1, 8, 16, 24, 32],
           ylabel = L"Weight Function, $f_r(j)$")

for (c_idx, c) in enumerate(obs_centers)
    lines!(fwd, weights[c_idx,:], label = "r = "*string(c))
end

axislegend(fwd, position = :rb)

save(destination*"_plot.pdf", fig)
