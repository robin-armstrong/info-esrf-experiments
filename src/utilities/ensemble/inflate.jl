using LinearAlgebra
using StatsBase

"""
Applies RTPP inflation with parameter `infl` to `X_f` and writes the result to `X_a`.
"""
function inflate!(X_f, X_a, infl)
    std_f = vec(std(X_f, dims = 2))
    std_a = vec(std(X_a, dims = 2))

    # setting up r such that r.*std_a = infl*std_f + (1 - infl)*std_a
    r   = std_f./std_a
    r .*= infl
    r .+= 1 - infl

    lmul!(Diagonal(r), X_a)
end
