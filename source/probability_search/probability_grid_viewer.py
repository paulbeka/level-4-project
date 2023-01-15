import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from networks.probability_finder import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions

ROOT_PATH = os.path.abspath(os.getcwd())


train, test = getPairSolutions(0.8, 1, 1)
model_path = os.path.join(ROOT_PATH, "models", "prob_test")
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

n_items = 1
n_iters = 20  # number of iterations on probability
testIterator = iter(test)

with torch.no_grad():
	for i in range(n_items):
		test_item = testIterator.next()
		result = model(test_item[0])
		print("TEST ITEM")
		print(test_item[0])
		print("MODIFICATION RESULT")
		print(result)
		print("ACTUAL")
		print(test_item[1])
		print("PREDICTED OUTPUT")
		plt.imshow(test_item[0][0] + result[0], cmap='gray', interpolation='nearest')
		plt.colorbar()
		plt.show()

		for i in range(n_iters):
			# iterate over new result to improve the final solution
			result = model(test_item[0][0] + result[0])