# RENAME THIS FILE TO recursive_analytics
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import pandas as pd

from networks.convolution_probability_network import ProbabilityFinder
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

			# plt.imshow(modeled_change[0], cmap='gray_r', interpolation='nearest')	
			# plt.colorbar()
			# plt.show()

			workState = addChangeVector(workState, modeled_change)

	return workState


def createTestingShipWithCellsMissing(ship, n_cells_missing):
	alive = np.argwhere(ship.numpy() == 1)	

	removed_cells = []
	for _ in range(min(len(alive)-1, n_cells_missing)):
		cell_being_removed = random.randint(0, len(alive)-1)
		removed_cells.append(tuple(alive[cell_being_removed][1:]))
		alive = np.delete(alive, cell_being_removed, axis=0)

	initialState = np.zeros_like(ship)
	initialState[0, alive[:, 1], alive[:, 2]] = 1
	initialState = torch.from_numpy(initialState)

	return initialState, removed_cells


def run_recursion_tests(pipeline, remove_counts_list, n_iters, n_items):
	testIterator = iter(train)	# use train iterator as there are more values

	with torch.no_grad():
		result_dict = {
			"n_iters" : [],
			"n_removed_cells" : [],
			"positive_scores" : [],
			"negative_scores" : [],
			"intactness_scores" : []
		}

		for n_removed_cells in remove_counts_list:
			for i in range(n_items):
				initialState, removed_cells = createTestingShipWithCellsMissing(testIterator.next()[1], n_removed_cells)

				result = itercycle(pipeline, initialState, n_iters)[0].numpy() # remove extra batch dimention used by neural net

				result_with_probs = [tuple(x) for x in list(np.argwhere(result > 0))]
				original_alive_cells = [tuple(x) for x in list(np.argwhere(initialState[0] == 1).reshape(-1, 2))]

				positive_scores = [result[x[0], x[1]] for x in removed_cells]
				negative_scores = [result[x[0], x[1]] for x in result_with_probs if not x in original_alive_cells]
				intactness_list = [result[x[0], x[1]] for x in result_with_probs if x in original_alive_cells]
				# prevent mean on empty slice
				if not positive_scores:	positive_scores = [0]
				if not negative_scores:	negative_scores = [0]
				if not intactness_list:	intactness_list = [0]

				result_dict["n_iters"].append(n_iters)
				result_dict["n_removed_cells"].append(n_removed_cells)
				result_dict["positive_scores"].append(np.mean())
				result_dict["negative_scores"].append(np.mean(negative_scores))
				result_dict["intactness_scores"].append(np.mean(intactness_list))

		return pd.DataFrame(result_dict)


def displayResults(result):
	iter_results = result.groupby(["n_iters"]).aggregate(np.mean)[["positive_scores", "negative_scores", "intactness_scores"]]
	removed_results = result.groupby(["n_removed_cells"]).aggregate(np.mean)[["positive_scores", "negative_scores", "intactness_scores"]]
	
	plt.plot(iter_results["positive_scores"], label="positive")
	plt.plot(iter_results["negative_scores"], label="negative")
	plt.plot(iter_results["intactness_scores"], label="intactness")
	plt.xlabel("Number of iterations")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()

	plt.plot(removed_results["positive_scores"], label="positive")
	plt.plot(removed_results["negative_scores"], label="negative")
	plt.plot(removed_results["intactness_scores"], label="intactness")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()


def run_tests(pipeline):
	MAX_ITER = 20	# the max amount of recursions
	MAX_REMOVE_COUNT = 10	# the maximum amount of cells to be removed

	test_n_iters = [i+1 for i in range(MAX_ITER)]
	remove_counts_list = [i+1 for i in range(MAX_REMOVE_COUNT)]
	test_n_spaceships = 25

	print(f"### N_SPACESHIPS = {test_n_spaceships} ###")
	n_iter_results = [run_recursion_tests(pipeline, remove_counts_list, n_iters, test_n_spaceships) for n_iters in test_n_iters]
	displayResults(pd.concat(n_iter_results))


train, test = getPairSolutions(0.8, 1, 1, "empty")

## LOADING MODELS
pipeline = []
pipe_names = ["3_epoch_10_iter_advanced_deconstruct"]

for item in pipe_names:
	model_path = os.path.join(ROOT_PATH, "models", item)
	model = ProbabilityFinder(1).double()
	model.load_state_dict(torch.load(model_path))
	model.eval()
	pipeline.append(model)

run_tests(pipeline)

