# Extended Life Cycle (ELC) Multiscale Particle Filter

This repository contains Python code implementing a multiscale Bayesian particle filter for simulating and estimating latent biological dynamics under the Extended Life Cycle (ELC) framework. The model captures feedback-driven evolutionary processes across nested temporal layers: development (individual life cycles), heredity (trait transmission within a population), and ecological dynamics (niche construction).

## Overview

The ELC framework treats the life cycle as the fundamental unit of evolutionary analysis. This implementation models:

- **Layer 1**: Individual life cycle dynamics (e.g., size, reproductive effort, survival probability) at time scale $t_1$ influenced by hereditary trait dynamis and the environmental dynamics
- **Layer 2**: Hereditary trait dynamics influenced by development (life cycle), population structures, and environment at time scale $t_2$
- **Layer 3**: Environmental dynamics shaped by feedback from individual life cycles via niche construction and optional hereditary traits at time scale $t_{3}

Each layer is formalized as a stochastic state-space model, enabling inference via particle filtering.

## Key Features

- Nested time scales with inverse $\epsilon$-scaling
- Individual-specific life history parameters and environmental niches
- Trait-based consumption and contribution influencing niche construction
- Support for dynamic environmental regime switching
- Visualization: true vs estimated states, RMSE plots, contribution/consumption plots, and animated GIFs

## Simulation Types
- Exclude environment from x2 dynamics, comment out + x3_effect in f_x2 function
- Exclude development from x3 dynamics, comment out 0.8 * combined_term in mu_t2 update
- Include x2 in x3 dynamics, set include_x2 = True in compute_phi function

## Dependencies

- Python 3.8+
- `numpy`
- `matplotlib`
- `os`, `pathlib`
- `matplotlib.animation` (Pillow backend required for saving GIFs)

Install dependencies with:

```bash
pip install numpy matplotlib pillow
