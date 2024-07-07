# RADNET

This is the repository for the RADNET architecture for polarization and electronic dielectric function predictions in periodic solids, and Raman intensities calculations.


## Folders structure

**figures/**: Contains codes and scripts to generate the data and figures of the paper

**radnet/**: Contains the code package to install to use the RADNET architecture

**scripts/**: Contains the various prewritten scripts used in the paper and based on the package. See the **Usage** section for more details on scripts. Every script can be called with `--help` to get an explanation of all input parameters.


## Installation
To install, simply `git clone` the repository and run
`pip install .`  (or `pip install -e .`  for local modifications)

## Scripts Usage
In all cases the `--help` argument will give a more detailed description of every input parameter.

`trainer.py`:
Trains a model instance from a `.h5` train dataset and a set of defined hyperparameters 

`inference.py`:
Uses a trained model instance and a `.h5` testing dataset (or the validation part of a training dataset) to evaluate the MAE and RMSE of the model

`predict_raman.py`:
Uses a trained model instance and a position input file (ase compatible) to predict all necessary values to the calculation of a Raman spectra. 

`predict_spectrum.py`:
Combines the quantities computed with `predict_raman.py` to output Raman intensities

