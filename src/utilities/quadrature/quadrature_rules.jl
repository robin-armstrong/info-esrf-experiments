using SpecialFunctions
using EllipticFunctions
using LinearAlgebra
using Random

# functions for computing quadrature rules in InFo-ESRF.

function golub_welsch(quadsize::Int)
    x   = 2*sqrt.(1 .- (2*(1:quadsize)).^(-2))
    G   = SymTridiagonal(zeros(quadsize), x.^(-1))
    eig = eigen(G)
    v   = 2*eig.vectors[1, :].^2
    lam = eig.values

    return lam, v
end

function setup_gaussian_quad(scale::Real, quadsize::Int)
    lam, v  = golub_welsch(quadsize)
    nodes   = tan.(.25*pi*(lam .+ 1)).^2
    weights = .5*v
    return nodes, weights
end

function elliptic_quadrature(scale::Real, quadsize::Int)
    s = zeros(quadsize)
    r = zeros(quadsize)
    
    k       = 1/sqrt(1 + scale)
    k_ellip = ellipk(1 - k^2)

    u = ((1:quadsize) .- .5)/quadsize
    s = imag(jellip("sn", k_ellip*u*im, m = k^2)).^2

    r   = real(jellip("cn", k_ellip*u*im, m = k^2))
    r .*= real(jellip("dn", k_ellip*u*im, m = k^2))
    r  *= k_ellip
    r  *= 2/(pi*quadsize)

    return s, r
end

function setup_elliptic_quad(scale::Real, quadsize::Int)
    s, r    = elliptic_quadrature(scale, quadsize)
    nodes   = s
    weights = r./(s .+ 1)

    return nodes, weights
end
