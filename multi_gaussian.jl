using LinearAlgebra
using LsqFit
using Statistics
using StatsBase

function multi_gaussian_pdf(x, p)
    num_gaussians = length(p) ÷ 2
    μ = @views p[1:num_gaussians]
    σ = @views p[num_gaussians+1:end]
    return [
        sum(
            exp(-0.5( (_x - μ[i])/σ[i])^2) / (√(2π) * σ[i])
            for i in 1:num_gaussians
        ) / num_gaussians
        for _x in x
    ]
end

function fit_multi_gaussian(samples, N)
    num_samples = length(samples)
    nbins = round(Int, sqrt(num_samples))

    histogram = normalize(fit(Histogram, samples; nbins), mode = :pdf)
    edges = histogram.edges[1]
    weights = histogram.weights
    centers = [0.5 * (edges[i] + edges[i+1]) for i in 1:length(edges)-1];

    μ_empirical = mean(samples)
    σ_empirical = stdm(samples, μ_empirical)

    μs = LinRange(μ_empirical - σ_empirical, μ_empirical + σ_empirical, N)
    σs = fill(σ_empirical / N, N)

    param = curve_fit(multi_gaussian_pdf, centers, weights, [μs; σs]).param

    μ = param[1:N]
    σ = param[N+1:end]
    return μ, σ
end

function sample_multi_gaussian(N, μs, σs)

end
