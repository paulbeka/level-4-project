# Readme

This project is a spaceship search algorithm for John Conway's Game of Life. It uses A* and two neural networks to guide the search for spaceships. 

### Requirements

* Python 3.9
* Packages: listed in `requirements.txt` 
* Tested on Windows 11 and latest working version of Linux Mint.

### Build Instructions/Steps

1. First, set up a new python environment for the project, `conda create --name gol_search`
2. When this is done, install the dependancies by running `pip install -r requirements.txt`

### Test steps

An initial test to see if the program is working correctly is by running `python probability_assisted_tree_search.py`. There should be some console output about some removed cells; your machine will probably make a lot of noise; and there will be output about the current iteration of the search. When the search is over, there will be a `Search over, found <x> spaceships` output.

