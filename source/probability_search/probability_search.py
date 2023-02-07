import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import random

from networks.convolution_probability_network import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions
from analytics import createTestingShipWithCellsMissing


ROOT_PATH = os.path.abspath(os.getcwd())

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) # root of the source file
sys.path.insert(1, PROJECT_ROOT)

from tools.rle_reader import RleReader  # in the tools class of source


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

		# currentState = nonStochasticProbabilityToStructure(currentState)
		# currentState = stochasticProbabilityToStructure(currentState)

		# plt.imshow(probability_matrix[0].detach(), cmap='gray_r', interpolation='nearest')	
		# plt.colorbar()
		# plt.show()

		# checkSpaceship(currentState)	# IMPLEMENT THE SPACESHIP CHECK

	# currentState = nonStochasticProbabilityToStructure(currentState)
	plt.imshow(currentState[0].detach(), cmap='gray_r', interpolation='nearest')	
	plt.colorbar()
	plt.show()


MAX_ITER = 5
MODEL_NAME = "conv_only_5x5_included"

model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

# Ship start with missing parts
rle_reader = RleReader()
filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
ships = rle_reader.getFileArray(filePath)

n_removed_cells = 1
initialState, removed_cells = createTestingShipWithCellsMissing(ships[77], n_removed_cells)
print(removed_cells)

plt.imshow(initialState[0].detach(), cmap='gray_r', interpolation='nearest')	
plt.colorbar()
plt.show()

# Random start
# initial_shape = (10, 19)
# initialState = torch.from_numpy(np.random.rand(*initial_shape).reshape(1, *initial_shape))

search(MAX_ITER, model, initialState)
