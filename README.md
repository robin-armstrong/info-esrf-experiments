This repository contains code which replicates the numerical experiments in R. Armstrong and I. Grooms, *Data Assimilation With An Integral-Form Ensemble Square-Root Filter* ([https://arxiv.org/abs/2503.00253](https://arxiv.org/abs/2503.00253)).

## Getting Started

To begin working with this code, run the following in a Bash terminal:
```
git clone https://github.com/robin-armstrong/info-esrf-experiments.git
cd info-esrf-experiments
$JULIA
```
where `JULIA` is the path to a Julia executable. Once in a Julia terminal, enter the following commands:
```
import Pkg; Pkg.activate(".")
Pkg.instantiate()
```
You will need to run `import Pkg; Pkg.activate(".")` every time you open Julia to work with this code. The `Pkg.instantiate()` command only needs to be run once; it installs project dependencies, a process which may take several minutes.

## Running Experiments
Experiments are located within the `src/experiments` directory, and are intended to be run from the top level. Each experiment should run out-of-the-box without additional configuration. To run a particular experiment, enter
```
include("src/experiments/<experiment-subdir>/run.jl")
```
in Julia, or enter
```
julia --project=. src/experiments/<experiment-subdir>/run.jl
```
from a shell at the top level.

Each `run.jl` file contains a variable called `destination`, defined near the top of the script, which controls where data is saved. The default setting looks like this:
```
destination = "src/experiments/<experiment-subdir>/<name>"
```
where `name` is an identifier for the particular instance of the experiment being run. Three different files are produced by `run.jl`:
1. `<name>_plot.pdf`, a plot of the experiment results.
2. `<name>_data.jld2`, the data being plotted.
3. `<name>_log.txt`, a file recording the parameter values used for this run of the experiment.

If you want to quickly run a plot without regenerating all the data (e.g., to change the scaling on an axis), then find the `plot_only` variable near the top of `run.jl` and set it to `true`. This will cause `run.jl` to look for a `*_data.jld2` file at the location specified by `destination`, and it will simply plot the contents of this file.