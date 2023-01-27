import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from networks.convolution_probability_network import ProbabilityFinder


ROOT_PATH = os.path.abspath(os.getcwd())


def nonStochasticProbabilityToStructure(probability_matrix):
	matrix = np.zeros_like(probability_matrix)
	alive = np.argwhere(probability_matrix[0] > 0.5).numpy().reshape(-1, 2)
	print(alive)
	if alive.size:
		matrix[0, alive[:, 0], alive[:, 1]] = 1
	return torch.from_numpy(matrix)


# implement this later
def stochasticProbabilityToStructure(probability_matrix):
	inverse_probs = 1 - probability_matrix
	new_matrix = np.random.choice([0, 1], probability_matrix.shape, [inverse_probs, probability_matrix])
	return torch.from_numpy(new_matrix).double()


def search(max_iter, model, currentState):

	for i in range(max_iter):
		probability_matrix = model(currentState)
		# currentState = nonStochasticProbabilityToStructure(currentState)
		currentState = stochasticProbabilityToStructure(currentState)

		# plt.imshow(currentState[0], cmap='gray_r', interpolation='nearest')	
		# plt.colorbar()
		# plt.show()

		# checkSpaceship(currentState)

	plt.imshow(currentState[0], cmap='gray_r', interpolation='nearest')	
	plt.colorbar()
	plt.show()


MAX_ITER = 50
MODEL_NAME = "3_epoch_10_iter_advanced_deconstruct"

model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

initialState = torch.from_numpy(np.random.rand(10, 10).reshape(1, 10, 10))

search(MAX_ITER, model, initialState)
