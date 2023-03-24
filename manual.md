## To run a search using the current models:
Inside of `probability_assisted_tree_search.py`, inside of the `if __name__ == '__main__':` if statement at the bottom of the program, call `search()`. To specifiy an initial grid, call `search(initialInput=<your grid>)`. The initial input needs to be a 2D torch grid of 1s and 0s. The number of iteration is a constant to be changed inside of the `search()` function itself.
Then execute `python probability_assisted_tree_search.py` to run the program. Make sure your environment is installed beforehand.

## To train the neural networks
The setup script is used to set up everything, and run the neural networks. Note that the `setup.sh` script can only be used in Linux, and there is no Windows version. It will create a new environment, install dependencies, and set up the training.

If you do want to run the training networks without the setup script:
1. If you have CUDA, skip this step. The two training scripts are `grid_probability_trainer.py` and `score_trainer.py`. Each have a CUDA check at the start of the program. Remove this check, and comment the lines 30 and 29 respectively.
2. To run training, simply execute `python grid_probability_trainer.py` for the probability change network, and `python score_trainer.py` for the scoring network.
3. If parameters need to be tuned, they are set as constants inside the trainers. To tune the data, the `probability_grid_dataloader.py` and `score_dataloader.py` scripts are inside the dataloaders file. 
