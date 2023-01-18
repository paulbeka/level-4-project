import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from networks.probability_finder import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions


ROOT_PATH = os.path.abspath(os.getcwd())


# ensures no values go above 1 or 0 when adding change 
# SIMPLIFY THIS CODE
def addChangeVector(change, target):
	result = (change + target).numpy()[0]
	remove_ones = np.argwhere(result > 1)
	result[remove_ones[:, 0], remove_ones[:, 1]] = 1
	remove_zeros = np.argwhere(result < 0)
	result[remove_zeros[:, 0], remove_zeros[:, 1]] = 0
	result = result[None, :]
	return torch.from_numpy(result) 


def itercycle(model_pipeline, initialState, n_iters):
	workState = initialState.detach().clone()
	for _ in range(n_iters):
		for model in model_pipeline:
			modeled_change = model(workState)

			plt.imshow(modeled_change[0], cmap='gray_r', interpolation='nearest')	
			plt.colorbar()
			plt.show()

			workState = addChangeVector(workState, modeled_change)

	return workState


train, test = getPairSolutions(0.8, 1, 1, "empty")

## LOADING MODELS
pipeline = []
pipe_names = ["10_epoch_empty"]

for item in pipe_names:
	model_path = os.path.join(ROOT_PATH, "models", item)
	model = ProbabilityFinder(1).double()
	model.load_state_dict(torch.load(model_path))
	model.eval()
	pipeline.append(model)


## PARAMETERS
n_iters = 1  # number of iterations on probability
scalar = 0.1
n_items = 1
testIterator = iter(test)

with torch.no_grad():
	for i in range(n_items):
		# INITIAL TEST STARTING STATES
		# use the test data to see if can rebuilt
		_, initialState = testIterator.next()
		alive = np.argwhere(initialState.numpy() == 1)	 # remove one cell from the structure
		print(f"The cell being removed is: {alive[0][1:]}")
		alive = np.delete(alive, 0, axis=0)
		initialState = np.zeros_like(initialState)
		initialState[0, alive[:, 1], alive[:, 2]] = 1
		initialState = torch.from_numpy(initialState)
		# initialState = torch.from_numpy(np.zeros((1, 10, 19)))	# 20x20 test matrix
		# initialState = torch.from_numpy(np.random.rand(1, 20, 20))

		result = itercycle(pipeline, initialState, n_iters)[0]
		result = result.numpy() # remove extra batch dimention used by neural net

		### TEST CODE ONLY
		bestOptionList = list(np.argwhere(result > 0))
		bestOptionList.sort(reverse=True, key=lambda x: result[x[0], x[1]])
		print([tuple(item) for item in bestOptionList])
		### ###

		### DISPLAY FINAL RESULT
		plt.imshow(result, cmap='gray_r', interpolation='nearest')	
		plt.title("Last iteration")
		plt.colorbar()
		plt.show()
