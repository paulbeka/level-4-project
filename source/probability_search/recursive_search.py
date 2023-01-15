import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from networks.probability_finder import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions


ROOT_PATH = os.path.abspath(os.getcwd())


def addChangeVector(change, target):
	result = change + target
	remove_ones = torch.argwhere(result > 1)
	result[remove_ones[:, 0], remove_ones[:, 1]] = 1
	remove_zeros = torch.argwhere(result < 0)
	result[remove_zeros[:, 0], remove_zeros[:, 1]] = 0
	return result 


train, test = getPairSolutions(0.8, 1, 1)
model_path = os.path.join(ROOT_PATH, "models", "prob_test")
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

n_items = 1
testIterator = iter(test)

## PARAMETERS
n_iters = 5  # number of iterations on probability
scalar = 0.1

with torch.no_grad():
	for i in range(n_items):
		result, solution = testIterator.next()

		for i in range(n_iters):
			changes = model(result)[0]
			result = addChangeVector(changes, result[0])
			result = result[None, :]

		plt.imshow(result[0], cmap='gray', interpolation='nearest')
		plt.title("Last iteration")
		plt.colorbar()
		plt.show()