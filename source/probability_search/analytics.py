import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from networks.convolution_probability_network import ProbabilityFinder
from networks.score_predictor import ScoreFinder
from dataloaders.probability_grid_dataloader import getPairSolutions

ROOT_PATH = os.path.abspath(os.getcwd()) # current root of the probability search

from tools.rle_reader import RleReader
from tools.testing import locationDifferencesBetweenTwoMatrixies, mockRatioDeconstruct
from tools.testing import createTestingShipWithCellsMissing, createTestingShipWithCellsAdded
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

			# workState = addChangeVector(workState, modeled_change)
			return modeled_change[0]	# workState[0] as it comes with a batch dimention


def run_recursion_tests(pipeline, extra_count_list, remove_counts_list, n_iters, n_items):
	with torch.no_grad():
		result_dict = {
			"n_iters" : [],
			"n_removed_cells" : [],
			"n_added_cells" : [],
			"positive_scores" : [],
			"negative_scores" : [],
			"intactness_scores" : [],
			"cohesion_scores" : [],
			"n_cells_missing_after_recursion" : [],
			"n_cells_extra_after_recursion" : [],
			"avg_scores_of_extra_cells" : [],
			"avg_scores_of_missing_cells" : []
		}

		for n_cells_extra in extra_count_list:
			for n_removed_cells in remove_counts_list:
				for i in range(n_items):
					test_ship = random.choice(ships)

					# INSTEAD, ADD RANDOM NOISE DECONSTRUCT
					# initialState, removed_cells = createTestingShipWithCellsMissing(test_ship, n_removed_cells)
					initialState, removed_cells, extra_cells = mockRatioDeconstruct(test_ship, n_removed_cells, n_cells_extra)

					result = itercycle(pipeline, initialState, n_iters)

					# take the absolute in order to also represent negative changes
					result_with_probs = [tuple(x) for x in list(np.argwhere(abs(result.clone().numpy()) > 0))]
					extra_cells = [tuple(x) for x in extra_cells]
					original_alive_cells = [tuple(x) for x in list(np.argwhere(initialState[0].numpy() == 1))]
					positive_scores = [result[x[0], x[1]].item() for x in removed_cells]
					negative_scores = [result[x[0], x[1]].item() for x in result_with_probs if not x in original_alive_cells]
					intactness_list = [result[x[0], x[1]].item() for x in result_with_probs if x in original_alive_cells]
					cohesion_score = [result[x[0], x[1]].item() for x in result_with_probs if x in extra_cells]

					# prevent mean on empty list
					if not positive_scores:	positive_scores = [0]
					if not negative_scores:	negative_scores = [0]
					if not intactness_list:	intactness_list = [0]
					if not cohesion_score: cohesion_score = [0]

					# TODO: ADD SOME SORT OF COHESION SCORE
					# YOU CAN START ADDING THE RESLULTS SOON AS THEY PROBS WONT CHANGE
					result_dict["n_iters"].append(n_iters)
					result_dict["n_removed_cells"].append(n_removed_cells)
					result_dict["n_added_cells"].append(n_cells_extra)
					result_dict["positive_scores"].append(np.mean(positive_scores))
					result_dict["negative_scores"].append(np.mean(negative_scores))
					result_dict["intactness_scores"].append(np.mean(intactness_list))
					result_dict["cohesion_scores"].append(np.mean(cohesion_score))

					result = result.numpy()
					newMatrix = initialState[0].numpy().copy()
					# the cells that were missing
					add_cells = np.argwhere(result > 0.15)
					if len(add_cells):					
						newMatrix[add_cells[:, 0], add_cells[:, 1]] = 1

					# the cells that were already there
					add_cells = np.argwhere((result < -0.1) & (result > -0.3))
					if len(add_cells):			
						newMatrix[add_cells[:, 0], add_cells[:, 1]] = 1

					remove_cells = np.argwhere(result < -0.7)
					if len(remove_cells):
						newMatrix[remove_cells[:, 0], remove_cells[:, 1]] = 0

					extra, missing = locationDifferencesBetweenTwoMatrixies(test_ship, newMatrix)
					result_dict["n_cells_missing_after_recursion"].append(len(missing))
					result_dict["n_cells_extra_after_recursion"].append(len(extra))
					extra_cell_scores = [result[x[0], x[1]] for x in extra]
					missing_cell_scores = [result[x[0], x[1]] for x in missing]
					if not extra_cell_scores: extra_cell_scores = [0]
					if not missing_cell_scores: missing_cell_scores = [0]
					result_dict["avg_scores_of_extra_cells"].append(np.mean(extra_cell_scores))
					result_dict["avg_scores_of_missing_cells"].append(np.mean(missing_cell_scores))

		return pd.DataFrame(result_dict)


def displayResults(result):
	# SHOULD YOU TRY THIS WITH NP.MAX TO SEE BETTER THRESHOLD VALUES?
	iter_results = result.groupby(["n_iters"]).aggregate(np.mean)
	removed_results = result.groupby(["n_removed_cells"]).aggregate(np.mean)
	added_results = result.groupby(["n_added_cells"]).aggregate(np.mean)
	
	plt.plot(iter_results["positive_scores"], label="positive")
	plt.plot(iter_results["negative_scores"], label="negative")
	plt.plot(iter_results["intactness_scores"], label="intactness")
	plt.plot(iter_results["cohesion_scores"], label="cohesion")
	plt.xlabel("Number of iterations")
	plt.ylabel("Score")
	plt.title("Number of iterations to score")
	plt.legend(loc="upper left")
	plt.show()
	plt.savefig("iterations_score_proabability_analysis")

	plt.plot(iter_results["n_cells_missing_after_recursion"], label="# Cells missing")
	plt.plot(iter_results["n_cells_extra_after_recursion"], label="# Cells extra")
	plt.xlabel("Number of iterations")
	plt.ylabel("Number of cells")
	plt.title("Number of iterations to cells missing/extra after search")
	plt.legend(loc="upper left")
	plt.savefig("iterations_n_cells_extra_missing_probability_analysis")
	plt.show()

	plt.plot(removed_results["positive_scores"], label="positive")
	plt.plot(removed_results["negative_scores"], label="negative")
	plt.plot(removed_results["intactness_scores"], label="intactness")
	plt.plot(removed_results["cohesion_scores"], label="cohesion")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.title("Number of removed cells to score")
	plt.legend(loc="upper left")
	plt.savefig("n_removed_cells_score_probability_analysis")
	plt.show()

	plt.plot(removed_results["n_cells_missing_after_recursion"], label="# Cells missing")
	plt.plot(removed_results["n_cells_extra_after_recursion"], label="# Cells extra")
	plt.xlabel("Number of cells removed from spaceship")
	plt.ylabel("Number of removed cells")
	plt.legend(loc="upper left")
	plt.title("Number of removed cells to cells missing/extra")
	plt.savefig("cells_removed_n_cells_extra_missing_probability_analysis")
	plt.show()

	plt.plot(added_results["positive_scores"], label="positive")
	plt.plot(added_results["negative_scores"], label="negative")
	plt.plot(added_results["intactness_scores"], label="intactness")
	plt.plot(added_results["cohesion_scores"], label="cohesion")
	plt.xlabel("Number of added cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.title("Number of added cells to score")
	plt.savefig("n_added_cells_score_probability_analysis")
	plt.show()

	plt.plot(added_results["n_cells_missing_after_recursion"], label="# Cells missing")
	plt.plot(added_results["n_cells_extra_after_recursion"], label="# Cells extra")
	plt.xlabel("Number of cells removed from spaceship")
	plt.ylabel("Number of added cells")
	plt.legend(loc="upper left")
	plt.title("Number of added cells to cells missing/extra")
	plt.savefig("cells_added_n_cells_extra_missing_probability_analysis")
	plt.show()

	plt.plot(added_results["avg_scores_of_extra_cells"], label="Extra cells avg score")
	plt.plot(added_results["avg_scores_of_missing_cells"], label="Missing cells avg score")
	plt.xlabel("Number of cells removed from spaceship")
	plt.ylabel("Change of probability (score)")
	plt.title("Number of added cells to score of missing/extra cells")
	plt.legend(loc="upper left")
	plt.savefig("cells_added_n_cells_extra_missing_probability_score_analysis")
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
	MAX_EXTRA_COUNT = 20

	test_n_iters = [i+1 for i in range(MAX_ITER)]
	remove_counts_list = [i+1 for i in range(MAX_REMOVE_COUNT)]
	extra_count_list = [i+1 for i in range(MAX_EXTRA_COUNT)]
	test_n_spaceships = 200

	print(f"### N_SPACESHIPS = {test_n_spaceships} ###")
	n_iter_results = [run_recursion_tests(pipeline, remove_counts_list, extra_count_list, n_iters, test_n_spaceships) for n_iters in test_n_iters]
	displayResults(pd.concat(n_iter_results))


### NEURAL NETWORK SCORE TESTING ###
def getMatrixScore(original_matrix, matrix):
	return torch.nn.MSELoss()(torch.from_numpy(original_matrix), torch.from_numpy(matrix)).item()


def runScoringTests(n_iters):

	model_name = "brandnewScoringNetworkEpoch_10"
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

	n_max_cells_missing = 50
	n_cells_missing_list = [i+1 for i in range(n_max_cells_missing)]
	for _ in range(n_iters):
		for i, ship in enumerate(ships):
			for n_cells_missing in n_cells_missing_list:
				# TEST THIS PROPERLY THERE MAY BE A MAJOR ISSUE WITH THIS
				testStructure, _ = createTestingShipWithCellsMissing(ship, n_cells_missing)
				testStructure, _ = createTestingShipWithCellsAdded(testStructure[0], n_cells_missing)
				score = -model(testStructure).item() + 0.14 # make the model close to the actual value
				# SEE IF THE RESULTS DIFFER IF THIS IS SWITCHED AROUND - THEY SHOULDNT NORMALLY I THINK
				# actualScore = getMatrixScore(testStructure.detach().numpy()[0], ship)
				actualScore = getMatrixScore(ship, testStructure.detach().numpy()[0])

				scores["score"].append(score)
				scores["actualScore"].append(actualScore)
				scores["trainedShip"].append(i > 800)
				scores["n_cells_missing"].append(n_cells_missing)

	data = pd.DataFrame(scores)

	cellsMissingTrained = data[data["trainedShip"] == True].groupby("n_cells_missing").aggregate(np.mean)
	cellsMissingUntrained = data[data["trainedShip"] == False].groupby("n_cells_missing").aggregate(np.mean)
	trainedShip = data.groupby("trainedShip").aggregate(np.mean)
	print(trainedShip)

	plt.plot(cellsMissingTrained["score"], label="Predicted Score")
	plt.plot(cellsMissingTrained["actualScore"], label="Actual Score")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.title("Trained ships cells missing to score")
	plt.savefig("cells_missing_to_score_trained")
	plt.show()

	plt.plot(cellsMissingUntrained["score"], label="Predicted Score")
	plt.plot(cellsMissingUntrained["actualScore"], label="Actual Score")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.title("Cells missing to score")
	plt.savefig("cells_missing_to_score_test")
	plt.show()


### SEARCH METHOD TESTING ###
# I SHOULD PROBABLY TEST STARTING WITH A SPACESHIP THAT IS FULLY COMPLETED BUT WITH A LOT OF ADDED EXTRA CELLS
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
		# MAYBE ADD SOME SORT OF SCORING METRIC?
		result_dict["cells_missing"].append(len(missing))
		result_dict["extra_cells"].append(len(extra))
		result_dict["found_ship"].append(int(len(extra) == 0 and len(missing) == 0)) 
	return result_dict


# Test the number of cells can be removed for search to work
def analyzeSearchMethodConvergence():
	# basically, i want to know if less max_depth but more iterations will make this algorithm better
	n_ships = 30	
	n_iter_list = [5]
	max_depth_list = [100]
	n_cells_removed_list = [i+1 for i in range(20)]
	ship_testing_list = random.sample(ships, k=n_ships)

	results_dict = {
		"n_iter" : [],
		"max_depth" : [],
		"n_cells_removed" : [],
		"mse_score" : [],
		"cells_missing" : [],
		"extra_cells" : [],
		"found_ship" : [],
	}

	total = len(n_iter_list) * len(max_depth_list) * n_ships * len(n_cells_removed_list)
	pbar = tqdm(desc="Analysing search algorithm: ", total=total)
	for n_iter in n_iter_list:
		for max_depth in max_depth_list:
			for ship in ship_testing_list:
				for n_cells_removed in n_cells_removed_list:
					damagedSpaceship, removed = createTestingShipWithCellsMissing(ship, n_cells_removed)
					results = search(
						n_iters=n_iter, 
						max_depth=max_depth, 
						initialInput=damagedSpaceship.clone(), 
						testing_data={
							"ship" : ship,
						})
					data = shipAfterSearchAnalysis(results, ship)
					results_dict["n_iter"] += [n_iter] * len(results)
					results_dict["max_depth"] += [max_depth] * len(results)
					results_dict["n_cells_removed"] += [n_cells_removed] * len(results)
					results_dict["mse_score"] += data["mse_score"]
					results_dict["cells_missing"] += data["cells_missing"]
					results_dict["extra_cells"] += data["extra_cells"]
					results_dict["found_ship"] += data["found_ship"] 

					pbar.update(1)

	pbar.close() # delete the progress bar

	results_pd = pd.DataFrame(results_dict)
	iter_grouped = results_pd.groupby(["n_iter"]).aggregate(np.mean)
	max_depth_grouped = results_pd.groupby(["max_depth"]).aggregate(np.mean)
	n_cells_removed_grouped = results_pd.groupby(["n_cells_removed"]).aggregate(np.mean)
	print(f"Number spaceships found: {len(results_pd[results_pd['found_ship'] == True])}")

	plt.plot(iter_grouped["cells_missing"], label="# cells missing")
	plt.plot(iter_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("# Cells removed")
	plt.ylabel("# cells ")
	plt.legend(loc="upper left")
	plt.savefig("number_of_cells_n_iter")
	plt.show()

	plt.plot(max_depth_grouped["mse_score"], label="MSE Score")
	plt.xlabel("Max depth")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.savefig("mse_score_max_depth")
	plt.show()

	plt.plot(max_depth_grouped["cells_missing"], label="# cells missing")
	plt.plot(max_depth_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("Max depth")
	plt.ylabel("# cells")
	plt.legend(loc="upper left")
	plt.savefig("number_of_cells_max_depth")
	plt.show()

	plt.plot(n_cells_removed_grouped["mse_score"], label="MSE Score")
	plt.xlabel("# cells removed")
	plt.ylabel("# of cells")
	plt.legend(loc="upper left")
	plt.savefig("mse_score_n_cells_removed")
	plt.show()

	plt.plot(n_cells_removed_grouped["cells_missing"], label="# cells missing")
	plt.plot(n_cells_removed_grouped["extra_cells"], label="# cells extra")
	plt.xlabel("# cells removed")
	plt.ylabel("# of cells")
	plt.legend(loc="upper left")
	plt.savefig("number_of_cells_n_cells_removed")
	plt.show()

	total_iterations = n_ships * len(n_iter_list) * len(max_depth_list)
	prob_reconstruction_aggregation = results_pd.groupby(["n_cells_removed"]).aggregate(np.sum)
	plt.plot(prob_reconstruction_aggregation["found_ship"] / total_iterations)
	plt.xlabel("# cells removed")
	plt.ylabel("Probability of reconstruction")
	plt.savefig("probability_of_reconstruction_against_damage")
	plt.show()

	n_iter_sum = results_pd.groupby(["n_iter"]).aggregate(np.sum)
	plt.plot(n_iter_sum["found_ship"])
	plt.xlabel("# of iterations")
	plt.ylabel("# of ships found")
	plt.savefig("n_iters_to_reconstruct")
	plt.show()


if __name__ == "__main__":
	rle_reader = RleReader()
	filePath = os.path.join(ROOT_PATH, "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)

	# if doing some training data testing:
	# ships = ships[:800]
	ships = ships[800:]

	# testing data:
	# ships = ships[800:]

	# run_ship_network_tests()
	# runScoringTests(100)
	analyzeSearchMethodConvergence()


# Current analytics available:
# - Network tests:
# --> Number of cells missing/extra for number of iterations
# --> 