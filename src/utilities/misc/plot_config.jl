# This variable controls fonts for plotting; please set it accordingly for your own system.
regfont = "/mnt/c/Windows/Fonts/times.ttf"

algnames  = ["base", "sensrf", "mp_getkf", "svd_getkf", "krylov_getkf", "info_esrf"]
alglabels = Dict("base" => "No DA",
                 "sensrf" => "Serial ESRF",
                 "mp_getkf" => "GETKF (mod.)",
                 "svd_getkf" => "GETKF (rSVD)",
                 "krylov_getkf" => "GETKF (Krylov)",
                 "info_esrf" => "InFo-ESRF")
algcolors  = Dict("base" => :purple,
                  "sensrf" => :red,
                  "mp_getkf" => :green,
                  "svd_getkf" => :blue,
                  "krylov_getkf" => :magenta,
                  "info_esrf" => :black)
algmarkers = Dict("base" => :x,
                  "sensrf" => :dtriangle,
                  "mp_getkf" => :circle,
                  "svd_getkf" => :diamond,
                  "krylov_getkf" => :utriangle,
                  "info_esrf" => :cross)
alglines   = Dict("base" => :dot,
                  "sensrf" => :dash,
                  "krylov_getkf" => :dashdot)
