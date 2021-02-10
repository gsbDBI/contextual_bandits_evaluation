#!/usr/bin/env bash
python setup.py develop
conda install -c r rpy2
conda install -c conda-forge r-devtools r-grf r-bh
git clone https://github.com/grf-labs/policytree.git && cd policytree
Rscript -e 'install.packages("r-package/policytree", repos = NULL, type = "source")'
