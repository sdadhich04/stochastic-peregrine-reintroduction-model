# Stochastic Peregrine Reintroduction Model

Computational biology coursework portfolio covering stochastic population viability, Leslie-matrix elasticity, Markov-chain channel dynamics, spike probability, and repressilator/network-motif ODE models.

## What's here

This repository is a post-course portfolio curation of AMATH 422/522 computational biology work. It contains cleaned, runnable Python code from the final project and from the recoverable code-heavy homework notebooks.

The final project implementation is in `src/falcon_pva.py`. It models peregrine falcon recolonization in Midwestern urban and cliff habitats with stochastic, stage-structured population dynamics:

- `projection_matrix(...)` builds stochastic juvenile, subadult, and adult transition matrices with habitat-specific survival and fecundity parameters.
- `density_dependence(...)` applies carrying-capacity effects to growth, survival, and fecundity terms.
- `demographic_step(...)` uses binomial survivor draws and Poisson juvenile production for demographic stochasticity.
- `disperse_juveniles(...)` moves juveniles across a patch network using distance-dependent dispersal probabilities and habitat-transition penalties.
- `simulate(...)` runs replicated, multi-year simulations for baseline, release, bridge-cliff, and lower-Midwest multi-cliff scenarios.
- `summarize(...)` reports expected minimum adult abundance, median time to colonization, and terminal adult abundance in the focal cliff patch.

`examples/run_scenarios.py` runs the final-project scenario sweep across:

- baseline urban-to-Southern-Illinois cliff dispersal,
- a bridge cliff between the urban source population and the Southern Illinois cliff,
- a larger lower-Midwest multi-cliff network,
- four release plans: no releases, and releases of 8, 16, or 24 juveniles every three years over the first decade.

The homework code and recovery notes are in `homeworks/`:

- HW1: literature-based modeling warmup; no substantive executable code was recovered.
- HW2: assignment/submission evidence exists, but no recoverable source notebook or script was found in the local archive.
- HW3: `homeworks/hw3_owl-elasticity/elasticity_analysis.py` computes Leslie-matrix growth, stable age distribution, reproductive value, and elasticity decomposition for an age-structured owl model.
- HW4: `homeworks/hw4_markov-dwell-times/markov_dwell_times.py` simulates a three-state ion-channel Markov chain, collapses closed/open states, estimates dwell-time distributions, and fits an exponential survival rate.
- HW5: `homeworks/hw5_ion-channel-spiking/spike_probability.py` computes stationary channel-state probabilities and exact/Monte Carlo spike probabilities across channel counts.
- HW6: `homeworks/hw6_repressilator-network-motifs/repressilator_network_motifs.py` simulates coupled repressilator ODEs and two-gene double-positive/double-negative regulatory motifs.

## What's original vs. course-provided

Original/coursework-derived work:

- The final project stochastic simulation logic, release schedules, bridge-habitat question, lower-Midwest patch network, and summary metrics are derived from group final-project notebooks in the local AMATH 422/522 course archive.
- The HW3-HW6 scripts are refactored from recovered homework notebooks into command-line Python modules.
- The portfolio version adds reusable functions, argument parsing, deterministic smoke-test seeds, and a public-source layout.

Course-provided or external material:

- Course PDFs, assignment prompts, Gradescope/Ed exports, presentation decks, report PDFs, and raw notebooks are intentionally not bundled.
- Biological framing and some parameter choices come from course/project context and the literature models studied in class, not from a private dataset included here.
- Final project collaborators: Jingyuan Cao, Yukai Lai, Siddharth Arvind, and Sparsh Dadhich.

## Known limitations / next steps

- HW1 is not represented as code because the recovered notebook is primarily written analysis rather than executable modeling code.
- HW2 is not represented as code because a source notebook/script was not recovered from the local archive. Reconstructing it from the assignment prompt would be new work, so it is left out.
- The original notebooks did not capture fixed random seeds for all reported results. The scripts use seeds for repeatable portfolio runs, but exact historical output reproduction is not guaranteed.
- The final project notebooks were prototype-style and hard-coded separate two-patch, bridge-cliff, and multi-cliff cases. This repo refactors that logic into a general patch-network simulator, but it has not been validated against every figure/table in the original course paper.
- Some biological parameter choices are encoded directly in `src/falcon_pva.py`. A next step would be loading parameter sets from documented CSV/JSON configs.
- The code has smoke-test coverage through direct script runs, but there are no automated unit tests yet.

## Running this code

Create a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the compact final-project scenario sweep:

```bash
python examples/run_scenarios.py
```

For a faster final-project smoke test:

```bash
python examples/run_scenarios.py --years 10 --replicates 20
```

Run the homework scripts:

```bash
python homeworks/hw3_owl-elasticity/elasticity_analysis.py
python homeworks/hw4_markov-dwell-times/markov_dwell_times.py --steps 20000
python homeworks/hw5_ion-channel-spiking/spike_probability.py --channels 40 --trials 20000
python homeworks/hw6_repressilator-network-motifs/repressilator_network_motifs.py --t-end 30
```

## License

MIT License. See `LICENSE`.
