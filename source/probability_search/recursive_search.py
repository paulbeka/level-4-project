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


def run_recursion(pipeline, remove_counts_list, n_iters, n_items):
	testIterator = iter(train)	# use train iterator as there are more values
	final_scores = {}

	with torch.no_grad():
		for n_removed_cells in remove_counts_list:
			removed_cell_result_list = []
			for i in range(n_items):
				initialState, removed_cells = createTestingShipWithCellsMissing(testIterator.next()[1], n_removed_cells)

				result = itercycle(pipeline, initialState, n_iters)[0].numpy() # remove extra batch dimention used by neural net

				result_with_probs = np.argwhere(result > 0)
				result_with_good_probs = np.argwhere(result > 0.5)

				original_alive_cells = np.argwhere(initialState == 1)
				negative_scores = [result[x[0], x[1]] for x in result_with_probs if not x in list(original_alive_cells)]
				intactness_list = [result[x[0], x[1]] for x in result_with_good_probs if x in list(original_alive_cells)]

				result_dict = {
					"positive_values" : [result[x[0], x[1]] for x in removed_cells],
					"negative_values" : negative_scores,
					"intactness_values" : intactness_list
				} 

				removed_cell_result_list.append(result_dict)


			avg_positive = sum([result[0] for result in removed_cell_result_list]) / n_items
			avg_negative = sum([result[1] for result in removed_cell_result_list]) / n_items


			final_scores.append((n_removed_cells, (avg_positive, avg_negative)))

	return final_scores


def run_tests(pipeline):
	MAX_ITER = 20
	MAX_REMOVE_COUNT = 10

	test_n_iters = [i+1 for i in range(MAX_ITER)]
	remove_counts_list = [i+1 for i in range(MAX_REMOVE_COUNT)]
	test_n_spaceships = 20

	result_dict = {
			"n_iters" : test_n_iters,
			"remove_counts_list" : remove_counts_list,

		}

	# STORE THIS STUFF IN PANDAS
	# DISPLAY COOL GRAPHS WITH REMOVE-INTEGRITY, and ITER-INTEGRITY
	# THEN USE THE NETWORK IN A TREE SEARCH

	print(f"### N_SPACESHIPS = {test_n_spaceships} ###")
	for n_iters in test_n_iters:
		print("-----")
		result_counts = run_recursion(pipeline, remove_counts_list, n_iters, test_n_spaceships)
		for result in result_counts:
			print(f"n_iters = {n_iters}, n_removed_cells = {result[0]}: positive average = {result[1][0]}, negative average = {result[1][1]}, intactness = {result[1][2]}")


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