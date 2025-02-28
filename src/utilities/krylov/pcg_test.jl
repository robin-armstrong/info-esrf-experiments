using LinearAlgebra
using Debugger
using Random

include("pcg.jl")
include("lmp.jl")

function run_pcg_test()
    rng = MersenneTwister(1)

    # defining the problem
    S  = exp10.(range(2, -2, 10))
    Q  = Matrix{Float64}(qr(randn(rng, 10, 10)).Q)
    A  = Q*Diagonal(S)*Q'
    B  = randn(rng, 10, 5)

    X0 = zeros(10, 5)
    R0 = B - A*X0

    # solving with un-preconditioned CG

    P0    = deepcopy(R0)
    prods = A*P0

    println("\nUN-PRECONDITIONED")
    println("-----------------")

    cg = pcg_initialize(X0, R0, P0, prods)

    while cg.iteration < 10
        println(cg.iteration,": R0 = ",norm(A*cg.sol - B))

        pcg_update_1!(cg, cg.testvectors)
        pcg_update_2!(cg, A*cg.testvectors)
    end

    println(cg.iteration,": R0 = ",norm(A*cg.sol - B))

    # re-doing the unpreconditioned solve but with an
    # in-place initialization.

    P0    = deepcopy(R0)
    prods = A*P0

    println("\nUN-PRECONDITIONED (in-place reinitialization)")
    println("-----------------")

    pcg_initialize!(cg, X0, R0, P0, prods)

    while cg.iteration < 10
        println(cg.iteration,": R0 = ",norm(A*cg.sol - B))

        pcg_update_1!(cg, cg.testvectors)
        pcg_update_2!(cg, A*cg.testvectors)
    end

    println(cg.iteration,": R0 = ",norm(A*cg.sol - B))

    # solving with preconditioned CG, first setting
    # up a limited-memory preconditioner that cuts
    # off the largest and smallest two eigenvalues.

    idx = [1, 2, 9, 10]
    W   = Q[:,idx]*inv(Diagonal(sqrt.(S[idx])))
    AW  = A*W
    G   = AW'*AW
    T1  = zeros(4, 5)
    T2  = zeros(4, 5)

    apply_lmp!(P0, R0, 1., W, AW, G, T1, T2)
    prods = A*P0

    println("\nLIMITED-MEMORY PRECONDITIONER")
    println("-----------------")

    cg = pcg_initialize(X0, R0, P0, prods)

    while cg.iteration < 10
        println(cg.iteration,": R0 = ",norm(A*cg.sol - B))

        apply_lmp!(prods, cg.testvectors, 1., W, AW, G, T1, T2)
        pcg_update_1!(cg, prods)
        pcg_update_2!(cg, A*cg.testvectors)
    end

    println(cg.iteration,": R0 = ",norm(A*cg.sol - B))
end

run_pcg_test()