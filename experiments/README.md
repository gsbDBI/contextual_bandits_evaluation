This directory contains scripts to run experiments using synthetic data and public classification datasets on [OpenML](https://www.openml.org), 
and make plots shown in the paper _Off-Policy Evaluation via Adaptive Weighting with Data from Contextual Bandits_.

## File description
- `script_synthetic.py`: run simulations on synthetic data described in the paper, and save results in `./results/` computed by different weighting schemes.
- `script_classification.py`: run simulations on classification data from OpenML, and save results in `./results/` computed by different weighting schemes.
- `statistics_synthetic.ipynb`: make plots of performances on synthetic data shown in the paper using saved results in `./results/`.
- `statistics_classification.ipynb`: make plots of performances on OpenML data shown in the paper using saved results in `./results/`.
- `intro_example.ipynb`: simulation of Example 1 in the paper.
- `plot_utils.py`: utility functions for making plots.



## Reproducibility 
To reproduce results on synthetic data shown in the paper, do
1. `python script_synthetic.py -s 1000` to run experiments and save results in `./results/`.
2. Open `statistics_synthetic.ipynb`, follow the instructions in the notebook to generate plots based on the saved results in `./results/`. 

To reproduce results on classification datasets shown in the paper, do
1. `python script_classification.py -s 100 -f {NameOfDataset}` to run experiments and save results in `./results/`.
2. Open `statistics_classification.ipynb`, follow the instructions in the notebook to generate plots based on the saved results in `./results/`. 


## Quick start for running simulations using synthetic data
Load packages
```python
import os
from adaptive.inference import analyze, aw_scores
from adaptive.experiment import *
from adaptive.ridge import *
from adaptive.datagen import *
from adaptive.saving import *
import openml
```
### 1. Collecting contextual bandits data
```python
K = 4 # Number of arms
p = 3 # Number of features
T = 7000 # Sample size
batch_sizes = [200] + [100] * 68 # Batch sizes
signal_strength = 0.5
config = dict(T=T, K=K, p=p, noise_form='normal', noise_std=1, noise_scale=0.5, floor_start=1/K, 
      bandit_model = 'RegionModel', floor_decay=0.8, dgp='synthetic_signal')

# Collect data from environment
data_exp, mus = simple_tree_data(T=T, K=K, p=p, noise_std=1, 
    split=0.5, signal_strength=signal_strength, noise_form='normal')
xs, ys = data_exp['xs'], data_exp['ys']
data = run_experiment(xs, ys, config, batch_sizes=batch_sizes)
yobs, ws, probs = data['yobs'], data['ws'], data['probs']
```

### 2. Evaluate optimal policy
```python
# Estimate muhat and gammahat
muhat = ridge_muhat_lfo_pai(data_exp['xs'], ws, yobs, K, batch_sizes)
gammahat = aw_scores(yobs=yobs, ws=ws, balwts=1 / collect(collect3(probs), ws),
                     K=K, muhat=collect3(muhat))
optimal_mtx = expand(np.ones(T), np.argmax(data_exp['muxs'], axis=1), K)
analysis = analyze(
                probs=probs,
                gammahat=gammahat,
                policy=optimal_mtx,
                policy_value=0.5,
            )
```


