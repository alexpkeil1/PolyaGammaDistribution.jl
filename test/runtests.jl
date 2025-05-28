using PolyaGammaDistribution
using Test
using Distributions: mean, var
using Random: GLOBAL_RNG

# Distributions.jl is willing to have tests that fail with some small probability
# even when the code is working, so let's just do that here.
@testset begin
    @test mean(PolyaGamma(2, 3.0)) ≈ 0.3017160845482888
    @test mean(PolyaGamma(1, 1.0)) ≈ 0.23105857863000487
    @test var(PolyaGamma(1, 1.0)) ≈ 0.03444664538852302
    @test var(PolyaGamma(1.0, 1.0)) ≈ 0.03444664538852302
end

@testset begin
   d = PolyaGamma(1, 3.0) 
   d2 = PolyaGamma(float(d.b), float(d.c)) 
   @test d.b == d2.b
   @test d.b !== d2.b
   x = 0.12  
   ntrunc = 200
   @test PolyaGammaDistribution.jacobi_logpdf(d.c / 2, 4.0 * x; ntrunc = ntrunc) + log(4) ≈ PolyaGammaDistribution.pg_logpdf(d.b, d.c, x; ntrunc=ntrunc)
end

@testset begin
    tol = 1e-3
    @test abs(mean([PolyaGammaDistribution.rtigauss(GLOBAL_RNG, 1.0) for _ in 1:10000]) - .372498) < .005
    @test abs(PolyaGammaDistribution.mass_texpon(0.0) - 0.5776972) < tol
    @test abs(PolyaGammaDistribution.mass_texpon(1.0) - 0.4605903) < tol
    @test abs(PolyaGammaDistribution.mass_texpon(2.0) - 0.2305365) < tol
end

@testset begin
    sigma_tol = 5
    d = PolyaGamma(1, 1.0)
    d2 = PolyaGamma(2.0, 2.0)
    analytic_mean = mean(d)
    analytic_var = var(d)
    analytic_mean2 = mean(d2)
    analytic_var2 = var(d2)
    nsamples = 10^6
    # standard error of the sample mean
    standard_err = sqrt(analytic_var / nsamples)
    standard_err2 = sqrt(analytic_var2 / nsamples)
    # standard error of the sample variance
    se_variance = sqrt(2*analytic_var^2/nsamples)
    se_variance2 = sqrt(2*analytic_var2^2/nsamples)
    @test abs(analytic_mean - mean(rand(d, nsamples))) < sigma_tol*standard_err
    @test abs(analytic_mean2 - mean(rand(d2, nsamples))) < sigma_tol*standard_err2
    @test abs(analytic_var -  var(rand(d, nsamples))) < sigma_tol*se_variance
    @test abs(analytic_var2 -  var(rand(d2, nsamples))) < sigma_tol*se_variance2
end
