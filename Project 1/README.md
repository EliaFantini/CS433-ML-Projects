# Project 1. Team pasta_balalaika

## Requirements

The easiest way to run the code is to install
Anaconda distribution (available for Windows, macOS and
Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/

Additional potential package versions are specified in the requirements.txt

## File description

- experiments/experiments_models.ipynb 

This Jupyter notebook contains our cross validation and hyperparameter experiments with different models

- experiments/experiments_preprocesing.ipynb

This Jupyter notebook contains our experiments with different preprocessing techniques

- experiments/generate_graphs.ipynb

Notebook that generates the graphs for the paper

- helper.py

Contains helper functions which were used for setting up our experiments  

- implementations.py

Contains 6 default required funcitons + additional minimization algorithms, and accoring loss funcitons 

- metrics.py

Contains our implementations of different metrics

- preprocessing.py

Contains methods for the preprocessing of data 

- run.py

Contains the code for reproducing our best submission file

- utils.py

Miscellaneous other functions, e.g. loading data, splitting it, etc..

- requirements.txt

File which includes package requirements for running the code

## Reproducing results

To reproduce our final submission results, simply run `python run.py` with according conda environment activated
