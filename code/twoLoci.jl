using Random, Distributions, Plots

fixProb(N, s) = s == 0.0 ? 1/N : (1 - exp(-2s)) / (1 - exp(-2N*s))

function analFix(N, uDel, uBen, sBen, mutFactor)
    deltaUdel = uDel * (mutFactor - 1)
    drift = fixProb(N, -deltaUdel)
    hitchhiking = (uBen / uDel) * fixProb(N, sBen)
    drift + hitchhiking
end

function simFix2Locus(N; uDel=1e-4, uBen=1e-6, sBen=0.1, sDel=-0.1, mutFactor=100, replicates=10000)
    fixations = 0
    for rep in 1:replicates
        freqMut = 1/N
        fixed = false
        extinct = false
        mutFitness = 0.0
        wtFitness = 0.0

        while !fixed && !extinct
            hasBenMutMut = rand() < mutFactor * uBen
            hasDelMutMut = rand() < mutFactor * uDel
            hasBenMutWt = rand() < uBen

            mutFitness += (hasBenMutMut ? sBen : 0) + (hasDelMutMut ? sDel : 0)
            wtFitness += hasBenMutWt ? sBen : 0

            wMut = max(1 + mutFitness, eps())
            wWt = max(1 + wtFitness, eps())

            meanW = freqMut * wMut + (1 - freqMut) * wWt
            probMut = clamp((freqMut * wMut) / meanW, 0.0, 1.0)
            freqMut = rand(Binomial(N, probMut)) / N

            fixed = freqMut == 1.0
            extinct = freqMut == 0.0
        end
        fixations += fixed
    end
    fixations / replicates
end

uDel, uBen, sBen, sDel, mutFactor = 1e-4, 1e-6, 0.1, -0.1, 100
popSizes = round.(Int, exp.(range(log(10), log(10000), length=75)))

analytic = [analFix(N, uDel, uBen, sBen, mutFactor) for N in popSizes]
simulated = [simFix2Locus(N, replicates=10000) for N in popSizes]

analyticNorm = popSizes .* analytic
simulatedNorm = popSizes .* simulated

plot(popSizes, analyticNorm, lw=2, label="Analytic approx.",
     xscale=:log10, yscale=:log10, xlabel="Population size (N)",
     ylabel="Normalized fixation probability (NP₍fix₎)",
     legend=:bottomright, ylim=(0.1, 10),
     xticks=([10, 100, 1000, 10000], [10, 100, 1000, 10000]),
     yticks=([0.1, 0.2, 0.5, 1, 2, 5, 10], [0.1, 0.2, 0.5, 1, 2, 5, 10]))

scatter!(popSizes, simulatedNorm, label="Stochastic simulations", color=:blue, ms=3)
hline!([1.0], linestyle=:dot, color=:black, label="Neutral expectation")
savefig("fig1J_2locus.png")
