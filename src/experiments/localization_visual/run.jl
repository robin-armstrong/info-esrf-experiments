using LinearAlgebra
using CairoMakie
using Random
using JLD2

include("../../utilities/ensemble/gaspari_cohn_univariate.jl")
include("../../utilities/misc/fprints.jl")
include("../../utilities/misc/rt.jl")

#############################################################################
################## SCRIPT PARAMETERS ########################################
#############################################################################

# data saving settings

plot_only   = false
destination = "src/experiments/localization_visual/localization"
readme      = "Visualizing model space covariance localization."

rng   = MersenneTwister(1)
N     = 100
k     = 20
alpha = .1

#############################################################################
################## DATA GENERATION ##########################################
#############################################################################

function run_localization_visual(destination, readme, rng, N, k, alpha)
    logstr  = "rng   = "*string(rng)*"\n"
    logstr *= "N     = "*string(N)*"\n"
    logstr *= "k     = "*string(k)*"\n"
    logstr *= "alpha = "*string(alpha)*"\n"
    logstr *= "\n"*readme*"\n"

    logfile = destination*"_log.txt"
    touch(logfile)
    io = open(logfile, "w")
    write(io, logstr)
    close(io)

    fprintln("\n"*logstr)

    C = ones(N, N)

    for i = 1:N
        for j = 1:N
            C[i, j] *= exp(-alpha*abs(i - j))
        end
    end

    C_sqrt = cholesky(C).L

    X = zeros(N, k)

    for j = 1:k
        X[:, j] = C_sqrt*randn(rng, N)
    end

    mu = (X*ones(k))/k
    X -= mu*ones(1, k)
    X /= sqrt(k - 1)

    L = zeros(N, N)

    for i = 1:N
        for j = 1:N
            L[i, j] = gaspari_cohn_univariate(abs(i - j), 2/alpha)
        end
    end

    @save destination*"_data.jld2" C X L
end

if !plot_only
    run_localization_visual(destination, readme, rng, N, k, alpha)
end

#############################################################################
################## PLOTTING #################################################
#############################################################################

@load destination*"_data.jld2" C X L

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (900, 300))

baseline = Axis(fig[1,1])
hidedecorations!(baseline)
heatmap!(baseline, rt(C), colormap = :vik, colorrange = (-1, 1))

ensemble = Axis(fig[1,2])
hidedecorations!(ensemble)
heatmap!(ensemble, rt(X*X'), colormap = :vik, colorrange = (-1, 1))

localized = Axis(fig[1,3])
hidedecorations!(localized)
heatmap!(localized, rt(L.*(X*X')), colormap = :vik, colorrange = (-1, 1))
Colorbar(fig[1,4], colormap = :vik, limits = (-1, 1))

save(destination*"_plot.pdf", fig)
