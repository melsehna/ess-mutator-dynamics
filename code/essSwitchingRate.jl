using Distributions, StatsBase, Random, Plots

function simulateInvasion(nGenerations, s, muRes, muMut, variance; seed=0)
    rng = MersenneTwister(seed)
    f = [0.4998, 0.4998, 0.0001, 0.0003]
    env = rand(rng, [0, 1])

    meanEnv = 20.0
    if variance == 0
        waitTimes = fill(Int(meanEnv), nGenerations)
    else
        shape = meanEnv^2 / variance
        scale = variance / meanEnv
        waitTimes = ceil.(Int, rand(rng, Gamma(shape, scale), nGenerations))
    end

    timePoints = []
    while sum(waitTimes) < nGenerations
        push!(waitTimes, rand(rng, Gamma(meanEnv^2 / variance, variance / meanEnv)))
    end
    for wt in waitTimes
        append!(timePoints, fill(env, Int(wt)))
        env = 1 - env
    end
    timePoints = timePoints[1:nGenerations]

    for t in 1:nGenerations
        f = updateFrequencies(f, s, muRes, muMut, timePoints[t])
    end

    return f[3] + f[4] > 1e-3
end

function updateFrequencies(freqs, s0, muRes, muMut, env)
    f0m, f1m, f0M, f1M = freqs
    s1 = s0

    if env == 0
        w = [1.0, 1.0 - s1, 1.0, 1.0 - s1]
    else
        w = [1.0 - s0, 1.0, 1.0 - s0, 1.0]
    end

    fitnesses = [f0m, f1m, f0M, f1M] .* w
    W = sum(fitnesses)
    fsel = fitnesses ./ W

    f0m′ = fsel[1] * (1 - muRes) + fsel[2] * muRes
    f1m′ = fsel[2] * (1 - muRes) + fsel[1] * muRes
    f0M′ = fsel[3] * (1 - muMut) + fsel[4] * muMut
    f1M′ = fsel[4] * (1 - muMut) + fsel[3] * muMut

    return [f0m′, f1m′, f0M′, f1M′]
end

function findESS(s, variance; trials=500, generations=100000)
    mu = 1e-2 * 10^(2*rand() - 1)
    for t in 1:trials
        multiplier = rand(Exponential(1.0))
        mu′ = mu * multiplier
        if simulateInvasion(generations, s, mu, mu′, variance)
            mu = mu′
        end
    end
    return mu
end

sVals = [0.01, 0.02, 0.05, 0.1, 0.5, 0.99]
varVals = [0, 5, 10, 20, 50, 100, 200, 400]

result = zeros(length(sVals), length(varVals))

for (i, s) in enumerate(sVals)
    for (j, var) in enumerate(varVals)
        println("s = $s, var = $var ...")
        result[i, j] = log10(findESS(s, var))
    end
end

plot(varVals, result', lw=2, markershape=:circle, xlabel="Environmental variance",
     ylabel="log₁₀(ESS switching rate)",
     label=["s=0.01" "s=0.02" "s=0.05" "s=0.1" "s=0.5" "s=0.99"],
     ylim=(-5, 0), legend=:bottomleft, grid=false)
hline!([-log10(20)], ls=:dash, label="1/n")
savefig("salatheFig1J.png")
