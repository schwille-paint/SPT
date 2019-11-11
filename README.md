# Single Particle Tracking based on DNA-PAINT

## Description
Custom module containing all relevant software for assessing the tracking performance of both immobilized and mobile targets.



## Table of contents
* [Installation](#installation)
* [Usage](#usage)
* [Remarks](#remarks)

## Installation

Setting up the conda environment:
1. Create a [conda][conda] environment ``conda create --name SPT python=3.7``
2. Activate the environment ``conda activate SPT``
3. Install necessary packages 
    * ``conda install h5py matplotlib numba numpy scipy pyqt pyyaml scikit-learn colorama tqdm=4.36.1 spyder pandas dask spyder fastparquet pytables jupyterlab``
    * ``pip install lmfit``


Installing the [picasso](https://github.com/jungmannlab/picasso) package: 

1. [Clone](https://help.github.com/en/articles/cloning-a-repository) the [picasso](https://github.com/jungmannlab/picasso) repository
2. Switch to the cloned folder ``cd picasso``
3. Install picasso into the environment ``python setup.py install``

Installing the [lbFCS](https://github.com/schwille-paint/lbFCS) package:

1. Leave the picasso directory ``cd ..``
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the [lbFCS](https://github.com/schwille-paint/lbFCS) repository
3. Switch to the cloned folder ``cd lbFCS``
4. Install lbFCS into the environment ``python setup.py install``

Installing the [trackpy](https://github.com/soft-matter/trackpy) package:

1. Leave the lbFCS directory ``cd ..``
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the [trackpy](https://github.com/soft-matter/trackpy) repository
3. Switch to the cloned folder ``cd trackpy``
4. Install trackpy into the environment ``python setup.py develop``

Installing the [pims](https://github.com/soft-matter/pims) package:

1. Leave the lbFCS directory ``cd ..``
2. [Clone](https://help.github.com/en/articles/cloning-a-repository) the [pims](https://github.com/soft-matter/pims) repository
3. Switch to the cloned folder ``cd pims``
4. Install pims into the environment ``python setup.py develop``

## Usage

### Remarks

