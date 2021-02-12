<h1 align="center">Adaptive Weighting in Contextual Bandits</h1>

Models for paper _Off-Policy Evaluation via Adaptive Weighting with Data from Contextual Bandits_.

<p align="center">
  Table of contents </br>
  <a href="#overview">Overview</a> •
  <a href="#development-setup">Development Setup</a> •
  <a href="#quickstart-with-model">Quickstart</a> 
</p>


# Overview

*Note: For any questions, please file an issue.*

Adaptive experimental designs can dramatically improve efficiency in randomized trials. But adaptivity also makes offline policy inference challenging. In the paper _Off-Policy Evaluation via Adaptive Weighting with Data from Contextual Bandits_, we propose a class of estimators that lead to asymptotically normal and consistent policy evaluation. This repo contains reproducible code for the results shown in the paper. 

We organize the code into two directories:
- [./adaptive](https://github.com/gsbDBI/contextual_bandits_evaluation/tree/main/adaptive) is a Python module for doing adaptive weighting developed in the paper.

- [./experiments](https://github.com/gsbDBI/contextual_bandits_evaluation/tree/main/experiments) contains python scripts to run experiments and make plots shown in the paper, including:
   - collecting contextual bandits data with a Thompson sampling agent;
   - doing off-line policy evaluation using collected data;
   - saving results and making plots. 

# Development setup

We recommend creating the following conda environment for computation.
```bash
conda create --name aw_contextual python=3.7
conda activate aw_contextual
source install.sh
```

# Quickstart with model

- To do adaptive weighting and reproduce results shown in the paper, please follow the instructions in [./experiments/README.md](https://github.com/gsbDBI/contextual_bandits_evaluation/blob/main/experiments/README.md).
- For a quick start, use
```bash
source activate aw_contextual
cd ./experiments/
python script_synthetic.py -T 1000
```


