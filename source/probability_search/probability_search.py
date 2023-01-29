import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from networks.convolution_probability_network import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions
from recursive_analytics import createTestingShipWithCellsMissing


ROOT_PATH = os.path.abspath(os.getcwd())


def nonStochasticProbabilityToStructure(probability_matrix):
	matrix = np.zeros_like(probability_matrix.detach())
	alive = np.argwhere(probability_matrix[0].detach().numpy() > 0.5)
	if alive.size:
		matrix[0, alive[:, 0], alive[:, 1]] = 1
	return torch.from_numpy(matrix)


def stochasticProbabilityToStructure(probability_matrix):
	inverse_probs = 1 - probability_matrix
	new_matrix = np.random.choice([0, 1], probability_matrix.shape, [inverse_probs, probability_matrix])
	return torch.from_numpy(new_matrix).double()


def search(max_iter, model, currentState):

	for i in range(max_iter):
		probability_matrix = model(currentState)
		currentState += probability_matrix 

		currentState = nonStochasticProbabilityToStructure(currentState)
		# currentState = stochasticProbabilityToStructure(currentState)

		# plt.imshow(currentState[0], cmap='gray_r', interpolation='nearest')	
		# plt.colorbar()
		# plt.show()

		# checkSpaceship(currentState)	# IMPLEMENT THE SPACESHIP CHECK

	plt.imshow(currentState[0], cmap='gray_r', interpolation='nearest')	
	plt.colorbar()
	plt.show()


MAX_ITER = 1
MODEL_NAME = "3_epoch_rand_addition_advanced_deconstruct"

model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

# Ship start with missing parts
train, test = getPairSolutions(0.8, 1, 1, "empty")
n_removed_cells = 1
testIterator = iter(train)
initialState, removed_cells = createTestingShipWithCellsMissing(testIterator.next()[1], n_removed_cells)

# Random start
# initial_shape = (10, 19)
# initialState = torch.from_numpy(np.random.rand(*initial_shape).reshape(1, *initial_shape))

search(MAX_ITER, model, initialState)
