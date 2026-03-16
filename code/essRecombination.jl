using Distributions, StatsBase, Random, Plots

function updateFrequenciesWithRecombination(freqs, s, muM, muMod, env, r)
    x1, x2, x3, x4 = freqs

    if env == 0
        w = [1.0, 1.0, 1.0 - s, 1.0 - s]
    else
        w = [1.0 - s, 1.0 - s, 1.0, 1.0]
    end

    D = x1 * x4 - x2 * x3
    W = sum(w .* freqs)

    x1′ = ((1 - muMod)*w[1]*(x1 - r*D) + muMod*w[3]*(x3 + r*D)) / W
    x2′ = ((1 - muM)*w[2]*(x2 + r*D) + muM*w[4]*(x4 - r*D)) / W
    x3′ = ((1 - muMod)*w[3]*(x3 + r*D) + muMod*w[1]*(x1 - r*D)) / W
    x4′ = ((1 - muM)*w[4]*(x4 - r*D) + muM*w[2]*(x2 + r*D)) / W

    return [x1′, x2′, x3′, x4′]
end

function simulateInvasion(nGen, s, muM, muMod, var, r; seed=0)
    rng = MersenneTwister(seed)
    f = [0.4998, 0.4998, 0.0001, 0.0003]
    env = rand(rng, [0, 1])

    meanEnv = 20.0
    if var == 0
        schedule = fill(Int(meanEnv), nGen)
    else
        shape = meanEnv^2 / var
        scale = var / meanEnv
        schedule = ceil.(Int, rand(rng, Gamma(shape, scale), nGen))
    end

    timepoints = []
    while sum(schedule) < nGen
        push!(schedule, rand(rng, Gamma(meanEnv^2 / var, var / meanEnv)))
    end
    for wt in schedule
        append!(timepoints, fill(env, Int(wt)))
        env = 1 - env
    end
    timepoints = timepoints[1:nGen]

    for t in 1:nGen
        f = updateFrequenciesWithRecombination(f, s, muM, muMod, timepoints[t], r)
    end

    return f[1] + f[3] > 1e-3
end

function findEss(s, var, r; trials=500, generations=100000)
    mu = 1e-2 * 10^(2*rand() - 1)
    for t in 1:trials
        multiplier = rand(Exponential(1.0))
        muPrime = mu * multiplier
        if simulateInvasion(generations, s, mu, muPrime, var, r)
            mu = muPrime
        end
    end
    return mu
end

function plotEssVsR()
    s = 0.1
    var = 20.0
    rVals = 0.0:0.05:0.5
    essVals = [findEss(s, var, r) for r in rVals]

    plot(rVals, log10.(essVals), xlabel="Recombination rate r", ylabel="log10(ESS switching rate)",
        grid=false, marker=:circle, lw=2)
    savefig("essRecombLiberman.png")
end

function plotEssVsVar()
    s = 0.1
    varVals = [0, 5, 10, 20, 50, 100, 200, 400]
    rVals = [0.0, 0.1, 0.3, 0.5]

    results = [log10.([findEss(s, var, r) for var in varVals]) for r in rVals]

    plot(varVals, results[1], label="r=0.0", lw=2, marker=:circle, xlabel="Environmental variance",
        ylabel="log10(ESS switching rate)", grid=false)
    for i in 2:length(rVals)
        plot!(varVals, results[i], label="r=$(rVals[i])", lw=2, marker=:circle)
    end
    hline!([-log10(20)], ls=:dash, label="1/n")
    savefig("essVarRecomb.png")
end

plotEssVsR()
plotEssVsVar()
