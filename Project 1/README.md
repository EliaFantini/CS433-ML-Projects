# Project 1. Team pasta_balalaika
AIcrowd ML Higgs
The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles
have mass. Its discovery at the Large Hadron Collider at CERN was announced in March 2013. In this project,
we applied machine learning techniques to actual CERN particle accelerator data to recreate the process of
“discovering” the Higgs particle. Physicists at CERN smash protons into one another at
high speeds to generate even smaller particles as by-products of the collisions. Rarely, these collisions can produce
a Higgs boson. Since the Higgs boson decays rapidly into other particles, scientists don’t observe it directly,
but rather measure its “decay signature”, or the products that result from its decay process. Since many decay
signatures look similar, we estimated the likelihood that a given event’s signature was the result of a
Higgs boson (signal) or some other process/particle (background). To do this, we implemented a pre-processing pipeline and different binary classification
techniques and compared their performance with hyperparameters tuning and cross validation.
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
