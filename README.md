# Caching Algorithm Simulator

Part of the Q4 2022 instance of the Delft University of Technology [Research
Project](https://github.com/TU-Delft-CSE/Research-Project), the concluding
thesis for the Bachelor's of Computer Science and Engineering.

A modular simulator for testing and benchmarking caching algorithms, primarily
for bipartite network algorithms. 
It was designed as part of a project to develop a bipartite version of the Online Mirror Descent caching policy.
The title of the related paper is "Integral Caching using Online Mirror Descent in a Networked Context".

JSON files allow for configuring benchmark runs, from network topology over
catalog size and time horizon to the seed used for all randomness involved. As
a result, runs can be re-created perfectly with only the JSON file. All data
presented in the paper can be re-created using the config files provided in the
`results` folder.

## Usage

In order to guarantee reproducibility, [Pipenv](https://pipenv.pypa.io/en/latest/) is used. 
This package can be installed through Hombrew (macOS) or a variety of Linux package managers.

To run a config, first change the path set in `run_from_config.py` to point
towards the desired config. Then, run `pipenv run python run_from_config.py`.
Note that paths are relative from the directory
from which `run_from_config.py` is executed. 
