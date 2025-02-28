using CairoMakie
using JLD2

include("../../utilities/quadrature/quadrature_rules.jl")
include("../../utilities/misc/plot_config.jl")
include("../../utilities/misc/fprints.jl")

#############################################################################
################## SCRIPT PARAMETERS ########################################
#############################################################################

# data saving settings

plot_only   = false
destination = "src/experiments/quadratures/quadratures"
readme      = "Comparing convergence rates of quadrature rules for the modified Kalman gain."

sigma_xh = 20.
sigma_hh = 10.
qrange   = 2:6

#############################################################################
################## DATA GENERATION ##########################################
#############################################################################

function run_quadrature_tests(destination, readme, sigma_xh, sigma_hh, qrange)
    logstr  = "sigma_xh = "*string(sigma_xh)*"\n"
    logstr *= "sigma_hh = "*string(sigma_hh)*"\n"
    logstr *= "qrange   = "*string(qrange)*"\n"
    logstr *= "\n"*readme*"\n"

    logfile = destination*"_log.txt"
    touch(logfile)
    io = open(logfile, "w")
    write(io, logstr)
    close(io)

    fprintln("\n"*logstr)

    kp = sigma_xh/(1 + sigma_hh + sqrt(1 + sigma_hh))   # scalar modified Kalman gain

    errs = Dict()

    for qrule in ["gaussian", "elliptic"]
        errs[qrule] = zeros(length(qrange))
    end

    for (q_idx, q) in enumerate(qrange)
        s, w = setup_gaussian_quad(1., q)
        khat = sum(sigma_xh*w./(s .+ 1 .+ sigma_hh))
        errs["gaussian"][q_idx] = abs(kp - khat)

        s, w = setup_elliptic_quad(2*sigma_hh, q)
        khat = sum(w.*sigma_xh./(s .+ 1 .+ sigma_hh))
        errs["elliptic"][q_idx] = abs(kp - khat)
    end

    @save destination*"_data.jld2" errs
end

if !plot_only
    run_quadrature_tests(destination, readme, sigma_xh, sigma_hh, qrange)
end

#############################################################################
################## PLOTTING #################################################
#############################################################################

@load destination*"_data.jld2" errs

CairoMakie.activate!(visible = false, type = "pdf")
fig = Figure(size = (400, 400), fonts = (; regular = regfont))

ax = Axis(fig[1,1],
          xlabel             = "Quadrature Size",
          xticks             = qrange,
          ylabel             = "Approximation Error",
          yscale             = log10,
          yminorgridvisible  = true,
          yminorticksvisible = true,
          yminorticks        = IntervalsBetween(10))

scatterlines!(ax, qrange, errs["gaussian"], marker = :utriangle, markersize = 15, label = "Gaussian")
scatterlines!(ax, qrange, errs["elliptic"], marker = :circle, markersize = 15, label = "Elliptic")
axislegend(ax)

save(destination*"_plot.pdf", fig)