#!/usr/bin/env bash
python setup.py develop
conda install -c r rpy2
conda install -c conda-forge r-devtools r-grf r-bh
Rscript -e 'install.packages("policytree", repos = "http://cran.us.r-project.org")'
