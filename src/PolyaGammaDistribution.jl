module PolyaGammaDistribution
using Distributions
using Random: AbstractRNG, GLOBAL_RNG, randexp
using StatsFuns: log1pexp
using SpecialFunctions: logabsgamma

const TRUNC = 0.64
const cutoff = 1 / TRUNC
const TERMS = 20


"""
    cosh(x) = (1+e⁻²ˣ)/(2e⁻ˣ)
    so logcosh(x) = log((1+e⁻²ˣ)) + x - log(2)
                  = x + log1pexp(-2x) - log(2)
"""
function logcosh(x::Real)
    x + log1pexp(-2x) - log(2)
end

"""
A Distribution containing the parameters ``b > 0`` and ``c`` for a Pólya-Gamma
distribution ``PG(b, c)``. Note that while in general ``b`` can be real,
some samplers implemented here only work for the integral case, and for non-integer numbers
an approximation method is used.
"""
struct PolyaGamma{T<:Real,U<:Real} <: ContinuousUnivariateDistribution
    b::T
    c::U
end

function jacobi_logpdf(z, x; ntrunc::Int)
    v = zero(x)
    for n = 0:ntrunc
        v += (iseven(n) ? 1 : -1) * acoef(n, x)
    end
    logcosh(z) - x * z^2 / 2 + log(v)
end


function lgamma(x)
    logabsgamma(x)[1]
end

"""
    The log coefficients of the infinite sum for the density of PG(b, 0).
    See Polson et al. 2013, section 2.3.
"""
function pg_logcoef(x, b, n)
    lgamma(n + b) - lgamma(n + 1) + log(2.0 * n + b) - log(2.0 * π * x^3) / 2.0 - (2.0 * n + b)^2 / (8.0 * x)
end
"""
   log density of the PG(b, 0) distribution.
    See Polson et al. 2013, section 2.3.
"""
function pg0_logpdf(x, b; ntrunc::Int)
    v = zero(x)
    for n = 0:ntrunc
        v += (iseven(n) ? 1 : -1) * exp(pg_logcoef(x, b, n))
    end
    (b - 1) * log(2) - lgamma(b) + log(v)
end

"""
    log density of the PG(b, c) distribution.
    See Polson et al. 2013, section 2.2 and equation (5).
"""
function pg_logpdf(b, c, x; ntrunc::Int)
    b * logcosh(c / 2) - x * c^2 / 2 + pg0_logpdf(x, b; ntrunc = ntrunc)
end

function Distributions.logpdf(d::PolyaGamma, x::Real; ntrunc::Int = TERMS)
    if d.b == 1
        return jacobi_logpdf(d.c / 2, 4 * x; ntrunc = ntrunc) + log(4)
    else
        return pg_logpdf(d.b, d.c, x; ntrunc = ntrunc)
    end
end
Distributions.pdf(d::PolyaGamma, x::Real; ntrunc::Int = TERMS) = exp(Distributions.logpdf(d, x; ntrunc = ntrunc))

"""
Analytically computes the mean of the given PG distribution, using the formula:

``
\frac{b}{2c} \tanh(\frac{c}{2})
``
"""
function Distributions.mean(d::PolyaGamma)
    (d.b / (2.0 * d.c)) * tanh(d.c / 2.0)
end

Distributions.rand(d::PolyaGamma) = rand(GLOBAL_RNG, d)
function Distributions.rand(rng::AbstractRNG, d::PolyaGamma)
    if (d.b === 1)
        #rpg_devroye(rng::AbstractRNG, num::T=1, n::T=1, z=0.0)
        res = rpg_devroye_1(rng, d.c)
    elseif typeof(d.b) <: Integer
        #rpg_devroye(rng::AbstractRNG, num::T=1, n::T=1, z=0.0)
        res = rpg_devroye(rng, d.b, d.c)[1]
    else
        #rpg_gammasum(rng::AbstractRNG, num::I=1, n::T=1.0, z::T=0.0, trunc::I=200)
        res = rpg_gammasum_1(rng, d.b, d.c, 200)
    end
    res
end


# Deriviation: https://stats.stackexchange.com/questions/122957/what-is-the-variance-of-a-polya-gamma-distribution

"""
Analytically computes the variance of the given PG distribution, using the formula

``
\\frac{b}{4c^3} (\\sinh(c) - c) \\sech(\\frac{c}{2})^2
``
"""
function Distributions.var(d::PolyaGamma)
    (d.b / (4 * d.c^3)) * (sinh(d.c) - d.c) * (sech(d.c / 2)^2)
end

function Distributions.shape(d::PolyaGamma)
    d.b
end

function Distributions.scale(d::PolyaGamma)
    d.c
end

# functions below are essentially translated from the BayesLogit R package

# cdf of Inverse Gaussian, already helpfully given to us
pigauss(x, μ, λ) = cdf(InverseGaussian(μ, λ), x)
"""
using Random
@time [rtigauss(MersenneTwister(123), 0.34) for _ in 1:10000];
"""
function randexp!(rng, w)
    for i in eachindex(w)
        w[i] = randexp(rng)
    end
    nothing
end

function rtigauss(rng::AbstractRNG, zin::Float64, r = TRUNC)
    z = abs(zin)
    μ = 1 / z
    x = r + 1
    if (μ > r)
        ee = zeros(Float64, 2)
        α = 0.0
        while rand(rng) > α
            randexp!(rng, ee)
            while ee[1]^2 > (2 * ee[2] / r)
                randexp!(rng, ee)
            end
            x = r / (1 + r * ee[1])^2
            α = exp(-0.5 * z^2 * x)
        end
    else
        while x > r
            λ = 1.0
            y = rand(rng, Normal(0.0, 1.0))^2
            x = μ + 0.5 * μ^2 / λ * y - 0.5 * μ / λ * sqrt(4 * μ * λ * y + (μ * y)^2)
            if rand(rng) > (μ / (μ + x))
                x = μ^2 / x
            end
        end
    end
    x
end

function mass_texpon(z::Float64, x = TRUNC)
    fz = π^2.0 / 8 + z^2.0 / 2.0
    b = sqrt(1.0 / x) * (x * z - 1.0)
    a = -1.0 * sqrt(1.0 / x) * (x * z + 1.0)
    x0 = log(fz) + fz * x
    SN = Normal(0.0, 1.0)
    xb = x0 - z + logcdf(SN, b)
    xa = x0 + z + logcdf(SN, a)

    qdivp = 4.0
    qdivp *= inv(π)
    qdivp *= (exp(xb) + exp(xa))
    qdivp += 1.0
    1.0 / qdivp
end

function acoef(n::I, x::T, r = TRUNC) where {I<:Int,T<:Real}
    n5 = float(n) + 0.5
    if (x > r)
        π * n5 * exp(-n5^2 * π^2.0 * x * 0.5)
    else
        (2.0 / π / x)^1.5 * π * n5 * exp(-2.0 * n5^2 / x)
    end
end

"""
Random draw from a Polya-Gamma distribution 

Pg(b=n,c=z) variable generation: single draw (num=1) from a single trial (n=shape=1) via the alternating series method of Devroye
pg 153, Devroye 1986

    - rng:	Random number generator (e.g. Random.MersenneTwister())
    - z:	Scale parameter. Parameter associated with tilting of the PG(b,0) distribution.

This is the sampler you want for a single draw from a single trial, as in with Bayesian logistic models

rpg_devroye_1(GLOBAL_RNG, 3.1)

acoef(0, 1.0, TRUNC)

"""
function rpg_devroye_1(rng::AbstractRNG, z)
    z::Float64 = abs(z) * 0.5
    ifz::Float64 = inv(0.125 * π^2.0 + 0.5 * z^2.0)
    x::Float64 = 0.0
    while true
        if rand(rng) < mass_texpon(z)
            x = TRUNC + randexp(rng) * ifz
        else
            x = rtigauss(rng, z)
        end
        s = acoef(0, x, TRUNC)
        y = rand(rng) * s
        n = 0
        while true
            n += 1
            if isodd(n)
                s -= acoef(n, x, TRUNC)
                y <= s && break
            else
                s += acoef(n, x, TRUNC)
                y > s && break
            end
        end
        y <= s && break
    end
    0.25 * x
end

"""
Random draws from a Polya-Gamma distribution 

Pg(b=n,c=z) variable generation: draws from a single trial (n=1, shape) via the alternating series method of Devroye
pg 153, Devroye 1986


    `rpg_devroye(rng, num, n, z)`

    `rpg_devroye(rng, n, z)` (for num=1)

    - rng:	Random number generator (e.g. Random.MersenneTwister())
    - num:	The number of random variates to simulate
    - n:	Shape parameter. Must be integer >= 1
    - z:	Scale parameter. Parameter associated with tilting of the PG(b,0) distribution
    
    ```{julia}
    rpg_devroye(GLOBAL_RNG, 10, 1, 0.0)
    ``
    using Random
@time [rpg_devroye(MersenneTwister(123), 1000, 1, .9) for _ in 1:1000];

"""
function rpg_devroye(rng::AbstractRNG, num::T, n::T, z = 0.0) where {T<:Int}
    x = zeros(Float64, num)
    @inbounds for i = 1:num
        x[i] += rpg_devroye(rng, n, z)
    end
    x
end

function rpg_devroye(rng::AbstractRNG, n::T, z) where {T<:Int}
    x = 0.0
    @inbounds for _ = 1:n
        x += rpg_devroye_1(rng, z)
    end
    x
end


"""
Random draws from a Polya-Gamma distribution 


Pg(b=n,c=z) variable generation (single draw, num=1) using a truncated infinite series of indepdendent draws froma Gamma(b,1) distribution

This function gets used when using `PG(b,c)` for a non-integer `b`

`rpg_gammasum(rng, n, z, nterms=PolyaGammaDistribution.TERMS)`

- rng:	Random number generator (e.g. Random.MersenneTwister())
- n:	Shape parameter. Must be > 0, but need not be integer
- z:	Scale parameter. Parameter associated with tilting of the PG(b,0) distribution (scale)
- nterms: The number of elements used to approximate the infinite convolution of gammas (higher is a better approximation, but slower).

Globally, you change the number of terms by setting PolyaGammaDistribution.TERMS to some different (integer) value. 

```{julia}
rpg_gammasum_1(GLOBAL_RNG, 1, 0.0, 200)
```
"""
function rpg_gammasum_1(rng::AbstractRNG, n::T, z::T, nterms::I = TERMS) where {I<:Int,T<:Real}
    ss = 0.0
    @inbounds for i = 1:nterms
        aa = (float(i) - 0.5) .^ 2.0 .* π .^ 2.0 .* 4.0 .+ z .^ 2.0
        ss += rand(rng, Gamma(n)) ./ aa
    end
    2.0 * ss
end


"""
Random draws from a Polya-Gamma distribution 


Pg(b=n,c=z) variable generation using a truncated infinite series of indepdendent draws froma Gamma(b,1) distribution

`rpg_gammasum(rng, num, n, z, nterms=PolyaGammaDistribution.TERMS)`

- rng:	Random number generator (e.g. Random.MersenneTwister())
- num:	The number of random variates to simulate
- n:	Shape parameter. Must be > 0, but need not be integer
- z:	Scale parameter. Parameter associated with tilting of the PG(b,0) distribution
- nterms: The number of elements used to approximate the infinite convolution of gammas (higher is a better approximation, but slower).

```{julia}
rpg_gammasum(GLOBAL_RNG, 1, 1, 0.0, 200)
```
using Random
@time [rpg_gammasum2(MersenneTwister(123), 1000, .8, .9) for _ in 1:100];
@time [rpg_gammasum(MersenneTwister(123), 1000, .8, .9) for _ in 1:100];
@time [rpg_gammasum2(MersenneTwister(123), 1000, .8, .9) for _ in 1:100];
@time [rpg_gammasum(MersenneTwister(123), 1000, .8, .9) for _ in 1:100];
"""
function rpg_gammasum(rng::AbstractRNG, num::I, n::T, z::T, nterms::I = TERMS) where {I<:Int,T<:Real}
    w = zeros(Float64, num)
    rpg_gammasum!(rng, w, n, z, nterms)
    w
end

function rpg_gammasum!(rng::AbstractRNG, w, n::T, z::T, nterms::I = TERMS) where {I<:Int,T<:Real}
    for i in eachindex(w)
        w[i] = rpg_gammasum_1(rng, n, z, nterms)
    end
end



"""
Random draw from a Polya-Gamma distribution 


Pg(b=n=1,c=z) variable generation: single draw (num=1) from a single trial (shape n=1) using alternative specification

`rpg_alt_1(rng, z)`

- rng:	Random number generator (e.g. Random.MersenneTwister())
- z:	Scale parameter. Parameter associated with tilting of the PG(b,0) distribution

"""
function rpg_alt_1(rng::AbstractRNG, z::T) where {T<:Real}
    α = 0.0
    x = 0.0
    while (rand(rng) > α)
        x = rpg_devroye_1(rng, 0.0)
        α = exp(-0.5 * (z * 0.5)^2.0 * x)
    end
    x
end


"""
Random draw from a Polya-Gamma distribution 


Pg(b=n=1,c=z) variable generation: draws from a single trial (shape n=1) using alternative specification

`rpg_alt(rng, num, z)`

- rng:	Random number generator (e.g. Random.MersenneTwister())
- num:	The number of random variates to simulate
- z:	Scale parameter. Parameter associated with tilting of the PG(b,0) distribution


```{julia}
rpg_alt(GLOBAL_RNG, 1, 0.3)
```

"""
function rpg_alt(rng::AbstractRNG, num::I, z::T) where {I<:Int,T<:Real}
    x = Array{Float64,1}(undef, num)
    @inbounds @simd for i = 1:num
        x[i] = rpg_alt_1(rng, z)
    end
    x
end

export PolyaGamma
end # module
