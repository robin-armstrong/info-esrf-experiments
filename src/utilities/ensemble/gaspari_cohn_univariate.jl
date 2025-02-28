"""
    gaspari_cohn_univariate(d, c)

Evaluates `G(d)`, where `d` is a scalar measure of distance and `G` is the Gaspari-Cohn localization function supported on `[-2c, 2c]`.
"""
function gaspari_cohn_univariate(d, c)
    x = abs(d)/c
    loc = 0.
    
    if(x <= 1.)
        loc = -.25*x^5 + .5*x^4 + .625*x^3 - (5/3)*x^2 + 1
    elseif(x <= 2.)
        loc = (1/12)*x^5 - .5*x^4 + .625*x^3 + (5/3)*x^2 - 5*x + 4 - 2/(3*x)
    end

    return loc
end
