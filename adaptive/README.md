This directory contains the Python module of adaptive weighting developed in the paper _Off-Policy Evaluation via Adaptive Weighting with Data from Contextual Bandits_.

To reproduce results presented in the paper, please go to directory [../experiments](https://github.com/gsbDBI/contextual_bandits_evaluation/tree/main/experiments) and follow the instructions in the README there. 

## File description
- `compute.py` contains helper functions to speed up computation. 
- `datagen.py` contains functions of creating data generating processes, including synthetic data and public classification datasets.
- `experiment.py` contains helper functions to run contextual bandits and multi-armed bandits.
- `inference.py` contains helper functions to do inference including computing scores and statistics of estimating policy values.
- `policy.py` contains PolicyTree utility functions.
- `region.py` contains implementation of a customized Thompson sampling agent, which firstly discretizes the covariate space using PolicyTree and then conduct TS sampling non-contextually within each subspace.
- `ridge.py` contains ridge regression utility functions.
- `saving.py` contains helping functions of different result-saving format. 
- `thompson.py` contains implementation of a linear Thompson sampling agent.
