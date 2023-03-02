import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import pandas as pd

from networks.convolution_probability_network import ProbabilityFinder
from networks.score_predictor import ScoreFinder
from dataloaders.probability_grid_dataloader import getPairSolutions

ROOT_PATH = os.path.abspath(os.getcwd()) # current root of the probability search

from tools.rle_reader import RleReader
from tools.testing import createTestingShipWithCellsMissing, locationDifferencesBetweenTwoMatrixies
from probability_assisted_tree_search import search


### NEURAL NETWORK CELL PREDICTION TESTING ###
# ensures no values go above 1 or 0 when adding change  SIMPLIFY THE CODE ITS BAD
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

			workState = addChangeVector(workState, modeled_change)

	return workState[0]	# workState[0] as it comes with a batch dimention


def run_recursion_tests(pipeline, remove_counts_list, n_iters, n_items):
	with torch.no_grad():
		result_dict = {
			"n_iters" : [],
			"n_removed_cells" : [],
			"positive_scores" : [],
			"negative_scores" : [],
			"intactness_scores" : [],
			"n_cells_missing_after_recursion" : [],
			"n_cells_extra_after_recursion" : []
		}

		for n_removed_cells in remove_counts_list:
			for i in range(n_items):
				test_ship = random.choice(ships)
				initialState, removed_cells = createTestingShipWithCellsMissing(test_ship, n_removed_cells)

				result = itercycle(pipeline, initialState, n_iters)

				result_with_probs = [tuple(x) for x in list(np.argwhere(result.numpy() > 0))]
				original_alive_cells = [tuple(x) for x in list(np.argwhere(initialState[0].numpy() == 1))]

				positive_scores = [result[x[0], x[1]].item() for x in removed_cells]
				negative_scores = [result[x[0], x[1]].item() for x in result_with_probs if not x in original_alive_cells]
				intactness_list = [result[x[0], x[1]].item() for x in result_with_probs if x in original_alive_cells]

				# prevent mean on empty slice
				if not positive_scores:	positive_scores = [0]
				if not negative_scores:	negative_scores = [0]
				if not intactness_list:	intactness_list = [0]

				result_dict["n_iters"].append(n_iters)
				result_dict["n_removed_cells"].append(n_removed_cells)
				result_dict["positive_scores"].append(np.mean(positive_scores))
				result_dict["negative_scores"].append(np.mean(negative_scores))
				result_dict["intactness_scores"].append(np.mean(intactness_list))

				result = result.numpy()
				newMatrix = np.zeros_like(result)
				threshold = np.mean(result)
				aboveHalf = np.argwhere(result > threshold)
				newMatrix[aboveHalf[:, 0], aboveHalf[:, 1]] = 1
				belowHalf = np.argwhere(result < threshold)
				newMatrix[belowHalf[:, 0], belowHalf[:, 1]] = 0
				extra, missing = locationDifferencesBetweenTwoMatrixies(test_ship, newMatrix)
				result_dict["n_cells_missing_after_recursion"].append(len(missing))
				result_dict["n_cells_extra_after_recursion"].append(len(extra))

		return pd.DataFrame(result_dict)


def displayResults(result):
	iter_results = result.groupby(["n_iters"]).aggregate(np.mean)
	removed_results = result.groupby(["n_removed_cells"]).aggregate(np.mean)
	
	plt.plot(iter_results["positive_scores"], label="positive")
	plt.plot(iter_results["negative_scores"], label="negative")
	plt.plot(iter_results["intactness_scores"], label="intactness")
	plt.xlabel("Number of iterations")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig("iterations_score_proabability_analysis")

	plt.plot(iter_results["n_cells_missing_after_recursion"], label="# Cells missing")
	plt.plot(iter_results["n_cells_extra_after_recursion"], label="# Cells extra")
	plt.xlabel("Number of iterations")
	plt.ylabel("Number of cells")
	plt.legend(loc="upper left")
	plt.savefig("iterations_n_cells_extra_missing_probability_analysis")
	plt.show()

	plt.plot(removed_results["positive_scores"], label="positive")
	plt.plot(removed_results["negative_scores"], label="negative")
	plt.plot(removed_results["intactness_scores"], label="intactness")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.savefig("n_removed_cells_score_probability_analysis")
	plt.show()

	plt.plot(removed_results["n_cells_missing_after_recursion"], label="# Cells missing")
	plt.plot(removed_results["n_cells_extra_after_recursion"], label="# Cells extra")
	plt.xlabel("Number of cells removed from spaceship")
	plt.ylabel("Number of cells")
	plt.legend(loc="upper left")
	plt.savefig("cells_removed_n_cells_extra_missing_probability_analysis")
	plt.show()

# TODO: ADD EXTRA CELLS THAT SHOULD NOT BE THERE (USE THE ADVANCED DECONSTRUCT METHOD)
def run_ship_network_tests():

	## LOADING MODELS
	pipeline = []
	pipe_names = ["5x5_included_20_pairs_epoch_4"]

	for item in pipe_names:
		model_path = os.path.join(ROOT_PATH, "models", item)
		model = ProbabilityFinder(1).double()
		model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
		model.eval()
		pipeline.append(model)

	MAX_ITER = 1	# the max amount of recursions
	MAX_REMOVE_COUNT = 20	# the maximum amount of cells to be removed

	test_n_iters = [i+1 for i in range(MAX_ITER)]
	remove_counts_list = [i+1 for i in range(MAX_REMOVE_COUNT)]
	test_n_spaceships = 20

	print(f"### N_SPACESHIPS = {test_n_spaceships} ###")
	n_iter_results = [run_recursion_tests(pipeline, remove_counts_list, n_iters, test_n_spaceships) for n_iters in test_n_iters]
	displayResults(pd.concat(n_iter_results))


### NEURAL NETWORK SCORE TESTING ###
def getMatrixScore(original_matrix, matrix):
	return torch.nn.MSELoss()(torch.from_numpy(original_matrix), torch.from_numpy(matrix)).item()


def runScoringTests(n_iters):

	model_name = "deconstructScoreOutputFile_2"
	model_path = os.path.join(ROOT_PATH, "models", model_name)
	model = ScoreFinder(1).double()
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()

	scores = {
		"score" : [],
		"actualScore": [],
		"trainedShip" : [],
		"n_cells_missing" : []
	}

	for _ in range(n_iters):
		for i, ship in enumerate(ships[800:]):
			n_cells_missing = random.randint(0, 100)
			testStructure, _ = createTestingShipWithCellsMissing(ship, n_cells_missing)
			score = model(testStructure).item()
			actualScore = getMatrixScore(testStructure.detach().numpy()[0], ship)

			scores["score"].append(score)
			scores["actualScore"].append(actualScore)
			scores["trainedShip"].append(i > 800)
			scores["n_cells_missing"].append(n_cells_missing)

	data = pd.DataFrame(scores)

	cellsMissing = data.groupby("n_cells_missing").aggregate(np.mean)
	trainedShip = data.groupby("trainedShip").aggregate(np.mean)
	print(trainedShip)

	plt.plot(cellsMissing["score"], label="Predicted Score")
	plt.plot(cellsMissing["actualScore"], label="Actual Score")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()


### SEARCH METHOD TESTING ###
def shipAfterSearchAnalysis(results, original_matrix):
	result_dict = {
		"mse_score" : [],
		"cells_missing" : [],
		"extra_cells" : [],
		"found_ship" : [],
	}
	for result in results:
		result = np.array(result.board[0]) # fetch the matrix assosiated with the fetched state
		result_dict["mse_score"].append(getMatrixScore(original_matrix, result))
		extra, missing = locationDifferencesBetweenTwoMatrixies(original_matrix, result)
		result_dict["cells_missing"].append(len(missing))
		result_dict["extra_cells"].append(len(extra))
		result_dict["found_ship"].append(len(extra) == 0 and len(missing) == 0)
	return result_dict


# Test the number of cells can be removed for search to work
def analyzeSearchMethodConvergence():

	n_ships = 5
	n_iter_list = [1]
	max_depth_list = [1, 50, 100]
	n_cells_removed_list = [1, 3, 5, 10]
	ship_testing_list = random.choices(ships, k=n_ships)

	results_dict = {
		"n_iter" : [],
		"max_depth" : [],
		"n_cells_removed" : [],
		"mse_score" : [],
		"cells_missing" : [],
		"extra_cells" : [],
		"found_ship" : [],
	}

	for n_iter in n_iter_list:
		for max_depth in max_depth_list:
			for ship in ship_testing_list:
				for n_cells_removed in n_cells_removed_list:
					damagedSpaceship, removed = createTestingShipWithCellsMissing(ship, n_cells_removed)
					results = search(n_iters=n_iter, max_depth=max_depth, initialInput=damagedSpaceship)
					data = shipAfterSearchAnalysis(results, ship)
					results_dict["n_iter"] += [n_iter] * len(results)
					results_dict["max_depth"] += [max_depth] * len(results)
					results_dict["n_cells_removed"] += [n_cells_removed] * len(results)
					results_dict["mse_score"] += data["mse_score"]
					results_dict["cells_missing"] += data["cells_missing"]
					results_dict["extra_cells"] += data["extra_cells"]
					results_dict["found_ship"] += data["found_ship"]

	results_pd = pd.DataFrame(results_dict)
	max_depth_grouped = results_pd.groupby(["max_depth"]).aggregate(np.mean)
	n_cells_removed_grouped = results_pd.groupby(["n_cells_removed"]).aggregate(np.mean)
	print(f"Number spaceships found: {len(results_pd[results_pd['found_ship'] == True])}")

	plt.plot(max_depth_grouped["mse_score"], label="MSE Score")
	# plt.plot(max_depth_grouped["cells_missing"], label="# cells missing")
	# plt.plot(max_depth_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("Max depth")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig("mse_score_max_depth")

	plt.plot(max_depth_grouped["cells_missing"], label="# cells missing")
	plt.plot(max_depth_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("Max depth")
	plt.ylabel("# cells")
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig("number_of_cells_max_depth")

	plt.plot(n_cells_removed_grouped["mse_score"], label="MSE Score")
	# plt.plot(n_cells_removed_grouped["cells_missing"], label="# cells missing")
	# plt.plot(n_cells_removed_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("# cells removed")
	plt.ylabel("# of cells")
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig("mse_score_n_cells_removed")

	plt.plot(n_cells_removed_grouped["cells_missing"], label="# cells missing")
	plt.plot(n_cells_removed_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("# cells removed")
	plt.ylabel("# of cells")
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig("number_of_cells_n_cells_removed")


if __name__ == "__main__":
	rle_reader = RleReader()
	filePath = os.path.join(ROOT_PATH, "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)

	run_ship_network_tests()
	# runScoringTests(100)
	# analyzeSearchMethodConvergence()


# Current analytics available:
# - Network tests:
# --> Number of cells missing/extra for number of iterations
# --> x`