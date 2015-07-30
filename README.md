# A Spiking Neural Model of the n-Back Task

This repository contains the model presented in the conference submission "A
Spiking Neural Model of the n-Back Task" and associated scripts.

## Files

* The actual n-back model is implemented in `nback/model.py`.
* Generation of n-back sequences is done with `nback/nbackgen.py`.
* The task parameters are set in `data/conf.py`.
* The plots were created with the IPython notebook `notebooks/Evaluation.ipynb`.

## Requirements

The code has been run with Python 2.7; Python 3 does not work. The following Python packages are required to run the model:

* [Nengo 2.0.1](https://github.com/nengo/nengo/releases/tag/v2.0.1)
* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)

Note that the model does not use the vanilla Nengo `spa` module, but a slightly
optimized version which is included in this repository as `nback/spaopt`.

For the evaluation notebook the following Python packages are needed:

* [IPython](http://ipython.org/)
* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Pandas](http://pandas.pydata.org/)
* [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/)

## Usage

`python nback/model.py --help` and `python nback/nbackgen.py --help` print usage
information for the respective files.

To generate all data shown in the paper a `dodo.py` file is provided. It can be
run with the `doit` command if the Python [doit](http://pydoit.org/) package is
installed.

