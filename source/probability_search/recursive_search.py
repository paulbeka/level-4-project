import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from networks.probability_finder import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions


ROOT_PATH = os.path.abspath(os.getcwd())


# ensures no values go above 1 or 0 when adding change vector
def addChangeVector(change, target):
	result = change + target
	remove_ones = torch.argwhere(result > 1)
	result[remove_ones[:, 0], remove_ones[:, 1]] = 1
	remove_zeros = torch.argwhere(result < 0)
	result[remove_zeros[:, 0], remove_zeros[:, 1]] = 0
	return result 


train, test = getPairSolutions(0.8, 1, 1)
model_path = os.path.join(ROOT_PATH, "models", "100_prob_network")
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

n_items = 1
testIterator = iter(test)

## PARAMETERS
n_iters = 3  # number of iterations on probability
scalar = 0.1

with torch.no_grad():
	for i in range(n_items):
		result, solution = testIterator.next()	# use the test data to see if can rebuilt
		result = result + solution
		# remove one cell from the structure
		alive = torch.argwhere(result == 1)
		print(alive[0][1:])
		result[alive[0, 1], alive[0, 2]] = 0
		# result = np.zeros((1, 20, 20))	# 20x20 test matrix
		# result = np.random.rand(1, 20, 20)

		for i in range(n_iters):
			changes = model(result)[0]

			# show the changes vector
			plt.imshow(changes, cmap='gray_r', interpolation='nearest')	
			plt.colorbar()
			plt.show()

			result = addChangeVector(changes, result[0])
			result = result[None, :]

		result = result[0].numpy() # remove extra batch dimention used by neural net

		bestOptionList = list(np.argwhere(result > 0))
		bestOptionList.sort(reverse=True, key=lambda x: result[x[0], x[1]])
		print(bestOptionList)

		plt.imshow(result, cmap='gray_r', interpolation='nearest')	
		plt.title("Last iteration")
		plt.colorbar()
		plt.show()