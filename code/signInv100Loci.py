import numpy as np
from numpy.random import poisson, binomial, multinomial
import matplotlib.pyplot as plt


class Lineage:
    def __init__(self, fitness, size, state, mutRate):
        self.fitness = fitness
        self.size = size
        self.state = np.array(state)
        self.mutRate = mutRate


def averageFitness(population):
    popN = sum(line.size for line in population.values())
    popW = sum(line.size * line.fitness for line in population.values())
    return popW / popN, popN


def assayMutationRate(population):
    popN = sum(line.size for line in population.values())
    mutN = sum(line.size for line in population.values() if line.mutRate > 1)
    return mutN / popN


def availSites(state):
    return [i for i, val in enumerate(state) if val == 0]


def mutatePopulation(population, Ub, sb, Ud, sd, mutMultiplier):
    newPopulation = dict()
    for key, line in population.items():
        mutations = poisson(line.mutRate * (Ud + Ub) * line.size)
        mutations = min(mutations, line.size)
        bMutations = binomial(mutations, Ub / (Ub + Ud))
        dMutations = mutations - bMutations
        openSites = availSites(line.state)
        if len(openSites) < bMutations + dMutations:
            continue
        mutationPositions = np.random.choice(openSites, bMutations + dMutations, replace=False)
        c = 0
        for _ in range(bMutations):
            newState = line.state.copy()
            newState[mutationPositions[c]] = sb
            c += 1
            key = tuple(newState)
            if key in newPopulation:
                newPopulation[key].size += 1
            else:
                newFit = 1 + np.sum(newState[1:])
                newPopulation[key] = Lineage(newFit, 1, newState, newState[0])
        for _ in range(dMutations):
            newState = line.state.copy()
            newState[mutationPositions[c]] = -sd
            c += 1
            key = tuple(newState)
            if key in newPopulation:
                newPopulation[key].size += 1
            else:
                newFit = 1 + np.sum(newState[1:])
                newPopulation[key] = Lineage(newFit, 1, newState, newState[0])
        newSize = line.size - bMutations - dMutations
        if newSize > 0:
            key = tuple(line.state)
            if key in newPopulation:
                newPopulation[key].size += newSize
            else:
                newPopulation[key] = Lineage(line.fitness, newSize, line.state.copy(), line.mutRate)
    return newPopulation


def wrightFisherReproduction(population, N0, Nnew, popw):
    probyList = [max(line.size / N0 * line.fitness / popw, 0.0) for line in population.values()]
    lineageList = list(population.values())
    newCountsList = multinomial(Nnew, probyList)
    newPopulation = dict()
    for i, count in enumerate(newCountsList):
        if count > 0:
            stateKey = tuple(lineageList[i].state)
            newPopulation[stateKey] = Lineage(
                lineageList[i].fitness, count, lineageList[i].state.copy(), lineageList[i].mutRate)
    return newPopulation


def simulate(popNi):
    initMutN = 1
    sb = 0.1
    sd = 0.1
    mutatorStrength = 100.0
    Ub = 1e-6
    Ud = 1e-4

    population = dict()
    wtState = np.zeros(100)
    wtState[0] = 1.0
    population[tuple(wtState)] = Lineage(1.0, popNi - initMutN, wtState, 1.0)
    mutState = wtState.copy()
    mutState[0] = mutatorStrength
    population[tuple(mutState)] = Lineage(1.0, initMutN, mutState, mutatorStrength)

    mutF = assayMutationRate(population)
    popw, popN = averageFitness(population)
    generations = 0

    while 0.0 < mutF < 1.0:
        popw, popN = averageFitness(population)
        population = wrightFisherReproduction(population, popN, popNi, popw)
        population = mutatePopulation(population, Ub, sb, Ud, sd, mutatorStrength)
        mutF = assayMutationRate(population)
        generations += 1

    return mutF == 1.0, generations, popw


def fixationProb(N, s):
    if s == 0:
        return 1 / N
    return (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * N * s))


def analyticFixation(N, Udel, Uben, sben, sdel, mutFactor):
    deltaUdel = Udel * (mutFactor - 1)
    drift = fixationProb(N, -deltaUdel)
    hitchhiking = (Uben / Udel) * fixationProb(N, sben)
    return drift + hitchhiking


if __name__ == '__main__':
    Udel = 1e-4
    Uben = 1e-6
    sben = 0.1
    sdel = -0.1
    mutFactor = 100

    Ns = np.round(np.exp(np.linspace(np.log(1), np.log(10000), 30))).astype(int)

    analytic = [analyticFixation(N, Udel, Uben, sben, sdel, mutFactor) for N in Ns]
    simulated = [simulate(N) for N in Ns]
    simFixProbs = [s[0] for s in simulated]

    analyticNorm = Ns * np.array(analytic)
    simulatedNorm = Ns * np.array([1.0 if s else 0.0 for s in simFixProbs])

    plt.figure(figsize=(8, 6))
    plt.plot(Ns, analyticNorm, color='black', label='Analytic approx.', linewidth=2)
    plt.scatter(Ns, simulatedNorm, color='royalblue', label='Stochastic simulations', s=50)
    plt.axhline(1.0, linestyle='dotted', color='gray', label='Neutral expectation')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Population size (N)')
    plt.ylabel('Normalized fixation probability (NP_fix)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1P.png', dpi=300)
    plt.show()
