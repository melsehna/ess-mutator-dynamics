using Random, Distributions, Plots

fixProb(N, s) = s == 0.0 ? 1/N : (1 - exp(-2s)) / (1 - exp(-2N*s))

function analFix(N, uDel, uBen, sBen, mutFactor)
    deltaUdel = uDel * (mutFactor - 1)
    drift = fixProb(N, -deltaUdel)
    hitchhiking = (uBen / uDel) * fixProb(N, sBen)
    drift + hitchhiking
end

function simFix(N; uDel=1e-4, uBen=1e-6, sBen=0.1, sDel=-0.1, mutFactor=100, replicates=100000)
    fixations = 0
    for rep in 1:replicates
        freqMut = 1/N
        fixed = false
        extinct = false
        mutFitness = 0.0
        wtFitness = 0.0

        while !fixed && !extinct
            benMutMut = rand(Poisson(mutFactor * uBen))
            delMutMut = rand(Poisson(mutFactor * uDel))
            benMutWt = rand(Poisson(uBen))

            mutFitness += benMutMut * sBen + delMutMut * sDel
            wtFitness += benMutWt * sBen

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

uDel, uBen, sBen, sDel = 1e-4, 1e-6, 0.1, -0.1
popSizes = round.(Int, exp.(range(log(10), log(100000), length=50)))
mutFactors = [1, 10, 25, 50, 75, 100]

resultsAnal = Dict()
resultsSim = Dict()

for mf in mutFactors
    println("Running simulations for mutFactor = $mf...")
    analytic = [analFix(N, uDel, uBen, sBen, mf) for N in popSizes]
    simulated = [simFix(N, uDel=uDel, uBen=uBen, sBen=sBen, sDel=sDel, mutFactor=mf,
                        replicates=N > 1000 ? 30000 : 10000) for N in popSizes]

    resultsAnal[mf] = popSizes .* analytic
    resultsSim[mf] = popSizes .* simulated
end

colors = [:darkorange, :dodgerblue, :forestgreen, :crimson, :gold, :deeppink]

plot(xscale=:log10, yscale=:log10,
     xlabel="Population size (N)",
     ylabel="Normalized fixation probability (NP₍fix₎)",
     legend=:outerright, ylim=(0.1, 10))

for (i, mf) in enumerate(mutFactors)
    plot!(popSizes, resultsAnal[mf], label="Analytic (x$mf)", color=colors[i], lw=2)
    scatter!(popSizes, resultsSim[mf], label="Sim (x$mf)", color=colors[i], ms=3)
end

hline!([1.0], linestyle=:dot, color=:black, label="Neutral expectation")
savefig("fig1J6.png")
