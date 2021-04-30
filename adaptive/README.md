This directory contains the Python module of adaptive weighting developed in the paper _Off-Policy Evaluation via Adaptive Weighting with Data from Contextual Bandits_.

To reproduce results presented in the paper, please go to directory [../experiments](https://github.com/gsbDBI/PolicyLearning/tree/main/scripts) and follow the instructions in the README there. 

## File description
- `bayesian.py`: function to do bayesion update.
- `compute.py`: helper functions to speed up computation. 
- `datagen.py`: functions of creating different data generating processes, including synthetic data and public classification datasets.
- `experiment.py`: functions to run contextual bandits.
- `inference.py`: functions to do inference including computing scores and statistics of estimating policy values.
- `policy.py`: PolicyTree utility functions.
- `region.py`: implementation of a customized Thompson sampling agent, which firstly discretizes the covariate space using PolicyTree and then conduct TS sampling non-contextually within each subspace.
- `ridge.py`: ridge regression utility functions.
- `saving.py`: functions to support different result-saving format. 
- `thompson.py`: implementation of a linear Thompson sampling agent.
