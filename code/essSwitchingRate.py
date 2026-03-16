import numpy as np
import matplotlib.pyplot as plt


def sampleEnvironmentWaitTime(meanWait, shape, rng):
    if shape > 1e5:
        return int(round(meanWait))
    scale = meanWait / shape
    val = rng.gamma(shape, scale)
    return max(1, int(round(val)))


def buildFlipTimes(meanWait, shape, maxGens, rng):
    flips = []
    currentT = 0
    while currentT < maxGens:
        dt = sampleEnvironmentWaitTime(meanWait, shape, rng)
        currentT += dt
        flips.append(currentT)
    return np.array(flips)


def selectionAndMutationStep(f0m, f0M, f1m, f1M, muM, muMod, s0, s1, envState):
    if envState == 0:
        w0m, w1m, w0M, w1M = 1.0, 1.0 - s1, 1.0, 1.0 - s1
    else:
        w0m, w1m, w0M, w1M = 1.0 - s0, 1.0, 1.0 - s0, 1.0

    sumFit = f0m * w0m + f1m * w1m + f0M * w0M + f1M * w1M
    if sumFit < 1e-30:
        sumFit = 1e-30

    f0mSel = (f0m * w0m) / sumFit
    f1mSel = (f1m * w1m) / sumFit
    f0MSel = (f0M * w0M) / sumFit
    f1MSel = (f1M * w1M) / sumFit

    f0mNew = f0mSel * (1 - muM) + f1mSel * muM
    f1mNew = f1mSel * (1 - muM) + f0mSel * muM
    f0MNew = f0MSel * (1 - muMod) + f1MSel * muMod
    f1MNew = f1MSel * (1 - muMod) + f0MSel * muMod

    return f0mNew, f0MNew, f1mNew, f1MNew


def simulatePopulation(muM, muMod, s0, s1, meanWait, shape,
                       maxGens=100000, burnIn=1000, initFreqs=None, rngSeed=42):
    rng = np.random.default_rng(rngSeed)
    flipTimes = buildFlipTimes(meanWait, shape, maxGens + burnIn + 1, rng)

    if initFreqs is None:
        f0m, f1m, f0M, f1M = 0.5, 0.5, 0.0, 0.0
    else:
        f0m = initFreqs.get('0m', 0.0)
        f1m = initFreqs.get('1m', 0.0)
        f0M = initFreqs.get('0M', 0.0)
        f1M = initFreqs.get('1M', 0.0)

    for g in range(burnIn + maxGens):
        envState = np.searchsorted(flipTimes, g) % 2
        f0m, f0M, f1m, f1M = selectionAndMutationStep(
            f0m, f0M, f1m, f1M, muM, muMod, s0, s1, envState)

    return (f0m, f0M, f1m, f1M)


def runPairwiseInvasionTrials(s0, s1, meanWait, shape,
                              nSteps=500, maxGens=100000,
                              burnIn=1000, rngSeed=42,
                              muRange=(1e-9, 0.1)):
    rng = np.random.default_rng(rngSeed)
    muM = 10.0 ** rng.uniform(np.log10(muRange[0]), np.log10(muRange[1]))
    popFreqs = {'0m': 0.5, '1m': 0.5, '0M': 0.0, '1M': 0.0}

    for step in range(nSteps):
        muMod = 10.0 ** rng.uniform(np.log10(muRange[0]), np.log10(muRange[1]))

        popBurn = {'0m': popFreqs['0m'], '1m': popFreqs['1m'], '0M': 0.0, '1M': 0.0}
        f0m, f0M, f1m, f1M = simulatePopulation(
            muM, 0.0, s0, s1, meanWait, shape,
            maxGens=burnIn, burnIn=0, initFreqs=popBurn,
            rngSeed=rng.integers(1e9))

        frac = 1e-4
        newF0m = max(0, f0m - frac / 2)
        newF1m = max(0, f1m - frac / 2)
        newF0M, newF1M = frac / 2, frac / 2
        total = newF0m + newF1m + newF0M + newF1M
        newF0m /= total; newF1m /= total
        newF0M /= total; newF1M /= total

        popInit = {'0m': newF0m, '0M': newF0M, '1m': newF1m, '1M': newF1M}

        f0mOut, f0MOut, f1mOut, f1MOut = simulatePopulation(
            muM, muMod, s0, s1, meanWait, shape,
            maxGens=maxGens, burnIn=0, initFreqs=popInit,
            rngSeed=rng.integers(1e9))

        freqM = f0MOut + f1MOut
        if freqM > 1e-3:
            muM = muMod
        popFreqs = {'0m': f0mOut, '0M': f0MOut, '1m': f1mOut, '1M': f1MOut}

    return muM


def replicateSalatheFigure1():
    sVals = [0.01, 0.1, 0.5, 0.99]
    shapeList = [1000, 100, 50, 20, 10, 5, 2, 1]
    nSteps = 500
    maxGens = 100_000

    results = {}
    for s in sVals:
        row = []
        for shp in shapeList:
            finalMu = runPairwiseInvasionTrials(
                s0=s, s1=s, meanWait=20.0, shape=shp,
                nSteps=nSteps, maxGens=maxGens,
                burnIn=1000, rngSeed=42 + int(shp * 10),
                muRange=(1e-9, 0.1))
            row.append(finalMu)
            print(f'Done: s={s}, shape={shp}, finalMu={finalMu:e}')
        results[s] = row

    fig, ax = plt.subplots()
    for s in sVals:
        xVals = [400.0 / shp for shp in shapeList]
        yVals = [np.log10(m) for m in results[s]]
        ax.plot(xVals, yVals, marker='o', label=f's={s}')

    ax.axhline(np.log10(1 / 20), color='gray', linestyle='--', label='1/n=1/20')
    ax.set_xlabel('Variance of environment')
    ax.set_ylabel('log10(ESS switching rate)')
    ax.legend()
    plt.savefig('salatheFig1.png')
    plt.show()


if __name__ == '__main__':
    replicateSalatheFigure1()
