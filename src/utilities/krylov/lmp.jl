"""
    apply_lmp!(Y, X, rho, W, AW, G, T1, T2)

Overwrites `Y` with `inv(P)*X`, where `P` is a limited-memory preconditioner for `A` 
defined by a positive scalar `rho` and a matrix `W`such that `W'*A*W = I`. This
preconditioner has the property that `inv(P)*A` has an eigenvalue cluster at `rho`,
and the largest (resp. smallest) eigenvalue of `inv(P)*A` is no larger (resp. smaller)
than that of `A`. The products `AW = A*W` and `G = AW'*AW` must be precomputed, along
with scratch-space arrays `t1` and `t2` of size `m x n` where `m = size(W, 2)` and
`n = size(X, 2)`.

For mathematical details on the preconditioner, see Tshimanga et al., Quarterly Journal
of the Royal Meteorological Society, 2008.
"""
function apply_lmp!(Y, X, rho, W, AW, G, T1, T2)
    mul!(T1, W', X)
    mul!(T2, AW', X)
    
    # Y = X - A*W*W'*X - W*W'*A*X + rho*W*W'*X
    copy!(Y, X)
    mul!(Y, W, T1, rho, 1)
    mul!(Y, AW, T1, -1, 1)
    mul!(Y, W, T2, -1, 1)

    # Y -= W*W'*A^2*W*W'*X
    mul!(T2, G, T1)
    mul!(Y, W, T2, 1, 1)
end
