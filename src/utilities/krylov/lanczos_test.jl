using LinearAlgebra
using Profile

include("lanczos.jl")

function run_lanczos_test()
    rng = MersenneTwister(1)

    # setting up the test matrix
    d = exp10.(range(2, -2, 100))
    V = Matrix{Float64}(qr(randn(rng, 100, 100)).Q)
    A = V*Diagonal(d)*V'

    # starting vectors

    B = randn(rng, 100, 5)

    # Lanczos process

    lanczos = lanczos_initialize(B, A*B)
    iters   = 20
    Q       = zeros(5, 100, iters)

    while lanczos.iteration < iters
        k = lanczos.iteration

        println("-------------------------")
        println("iter   = ",k)
        println("alphas = ",lanczos.alpha)
        println("betas  = ",lanczos.beta)
        
        Q[:, :, k] = lanczos.Q'

        Q_k   = [Q[i, :, 1:k] for i = 1:5]
        evd   = [eigen(Hermitian(Q_k[i]'*A*Q_k[i])) for i = 1:5]
        lmins = [evd[i].values[1] for i = 1:5]
        lmaxs = [evd[i].values[end] for i = 1:5]

        println("lmins  = ",lmins)
        println("lmaxs  = ",lmaxs)

        lanczos_update!(lanczos, A*lanczos.testvectors)
    end
end

run_lanczos_test()
