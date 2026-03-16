# ess-mutator-dynamics

Evolutionary simulations of mutator dynamics and ESS switching rates in fluctuating environments. Models based on Raynes et al. (2018), Salathe et al. (2009), and Liberman et al. (2011).

## Models

### Sign Inversion in Mutator Fixation (Raynes et al., 2018)
Wright-Fisher simulations showing the U-shaped relationship between population size and mutator fixation probability. At small N, mutators fix by drift; at large N, beneficial hitchhiking dominates; at intermediate N, deleterious load suppresses fixation.

- `code/signInv6.jl` — Multi-locus Julia simulation across mutator strengths (1x–100x)
- `code/twoLoci.jl` — Simplified 2-locus Julia version
- `code/signInv100Loci.py` — Full 100-locus Python simulation with lineage tracking

### ESS Switching Rate (Salathe et al., 2009)
Two-locus modifier model for the evolutionarily stable switching rate in fluctuating environments. A resident switching rate is repeatedly challenged by invading mutant rates via pairwise invasion trials.

- `code/essSwitchingRate.jl` — Julia implementation
- `code/essSwitchingRate.py` — Python implementation

### Recombination and ESS (Liberman et al., 2011)
Extension of the Salathe model incorporating recombination between the major and modifier loci. Shows that recombination disrupts linkage and suppresses the evolution of adaptive switching.

- `code/essRecombination.jl` — ESS vs recombination rate and ESS vs environmental variance for different r values

### Memes (Cultural Evolution)
Discrete-time recursion model for meme spread: `f(t+1) = f(t)(1 - gamma) + alpha * f(t)^2 * (1 - f(t))`. Plots equilibria and their stability.

- `code/memes.py`

## References

- Raynes, Y., Gazzara, M. R., & Sniegowski, P. D. (2018). Mutator dynamics in sexual and asexual experimental populations of yeast. *PNAS*, 115(13), 3422–3427.
- Salathe, M., Van Cleve, J., & Feldman, M. W. (2009). Evolution of stochastic switching rates in asymmetric fitness landscapes. *Genetics*, 182(4), 1159–1164.
- Liberman, U., Van Cleve, J., & Feldman, M. W. (2011). Stochastic effects in the evolution of mutation rates in fluctuating environments. *Genetics*, 187(3), 837–846.
