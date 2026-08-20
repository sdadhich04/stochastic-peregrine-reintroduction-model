# Peregrine Falcon Population Viability Simulation

Stochastic population-viability modeling for peregrine falcon reintroduction strategies, with stage-structured dynamics, juvenile dispersal, density dependence, and bridge-habitat experiments.

## What's here

This repository is a post-course portfolio curation of an AMATH 422/522 computational biology final project by Jingyuan Cao, Yukai Lai, Siddharth Arvind, and Sparsh Dadhich. The code models peregrine falcon recolonization in Midwestern urban and cliff habitats, based on a spatially explicit population viability analysis from the biological conservation literature.

The core implementation is in `src/falcon_pva.py`:

- `projection_matrix(...)` builds a stochastic Leslie-style transition matrix for juvenile, subadult, and adult stages, with habitat-specific survival and fecundity parameters.
- `density_dependence(...)` applies carrying-capacity effects to growth, survival, and fecundity terms.
- `demographic_step(...)` uses binomial survivor draws and Poisson juvenile production to represent demographic stochasticity.
- `disperse_juveniles(...)` moves juveniles across a patch network using distance-dependent dispersal probabilities and habitat-transition penalties.
- `simulate(...)` runs replicated, multi-year stochastic simulations for baseline, release, bridge-cliff, and lower-Midwest multi-cliff scenarios.
- `summarize(...)` reports expected minimum adult abundance, median time to colonization, and terminal adult abundance in the focal cliff patch.

`examples/run_scenarios.py` runs a compact scenario sweep:

- baseline urban-to-Southern-Illinois cliff dispersal,
- a bridge cliff between the urban source population and the Southern Illinois cliff,
- a larger lower-Midwest multi-cliff network,
- four release plans: no releases, and releases of 8, 16, or 24 juveniles every three years over the first decade.

## What's original vs. course-provided

Original/coursework-derived work:

- The stochastic simulation logic, scenario modeling, release schedules, bridge-habitat question, and summary metrics are derived from group final-project notebooks in the local AMATH 422/522 course archive.
- The portfolio version refactors the notebook prototypes into reusable Python functions with a scriptable demo runner.

Course-provided or external material:

- Course PDFs, assignment prompts, platform exports, and presentation/report PDFs are intentionally not bundled.
- Parameters and biological framing were taken from the project literature model and course project context, not from a private dataset included in this repository.
- Final project collaborators: Jingyuan Cao, Yukai Lai, Siddharth Arvind, and Sparsh Dadhich.

## Known limitations / next steps

- The original notebooks did not capture fixed random seeds for their reported results, so exact historical output reproduction is not guaranteed. The demo runner uses a seed for repeatable portfolio runs.
- The original project notebooks were prototype-style and hard-coded separate two-patch, bridge-cliff, and multi-cliff cases. This repo refactors that logic into a general patch-network simulator, but it has not been validated against every figure/table in the original course paper.
- Some biological parameter choices are encoded directly in `DEFAULT_HABITATS` and scenario definitions. A next step would be loading parameter sets from documented CSV/JSON configs.
- The code has been refactored from notebooks into a reusable simulator, but there are no automated unit tests yet. A next step would be adding tests around probability normalization, release scheduling, and summary metrics.

## Running this code

Create a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the compact scenario sweep:

```bash
python examples/run_scenarios.py
```

For a faster smoke test:

```bash
python examples/run_scenarios.py --years 10 --replicates 20
```

For a heavier run closer to the original notebooks:

```bash
python examples/run_scenarios.py --years 50 --replicates 1000
```

## License

MIT License. See `LICENSE`.
