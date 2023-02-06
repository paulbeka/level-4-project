import numpy as np
import torch
import os
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd

from networks.convolution_probability_network import ProbabilityFinder
from networks.score_predictor import ScoreFinder
from dataloaders.probability_grid_dataloader import getPairSolutions


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) # root of the source file
sys.path.insert(1, PROJECT_ROOT)

ROOT_PATH = os.path.abspath(os.getcwd()) # current root of the probability search


from tools.rle_reader import RleReader  # in the tools class of source


# ensures no values go above 1 or 0 when adding change  SIMPLIFY THE CODE ITS SHIT
def addChangeVector(change, target):
	result = (change + target).numpy()[0]
	remove_ones = np.argwhere(result > 1)
	result[remove_ones[:, 0], remove_ones[:, 1]] = 1
	remove_zeros = np.argwhere(result < 0)
	result[remove_zeros[:, 0], remove_zeros[:, 1]] = 0
	result = result[None, :]
	return torch.from_numpy(result) 


# turn a flattened output grid into a grid
def modelOutputToGridAndScore(input_shape, model_output):
	flattened_grid = model_output[0, :model_output.shape[1]-1]
	grid = flattened_grid.reshape(input_shape)
	score = model_output[0, -1].item()
	return (grid[0], score)


def itercycle(model_pipeline, initialState, n_iters):
	workState = initialState.detach().clone()
	startShape = workState.shape
	score = 0
	for _ in range(n_iters):
		for model in model_pipeline:
			modeled_change = model(workState)
			# modeled_change, score = modelOutputToGridAndScore(startShape, modeled_change)

			# plt.imshow(-modeled_change, cmap='gray_r', interpolation='nearest')	
			# plt.colorbar()
			# plt.show()

			workState = addChangeVector(workState, modeled_change)
			# workState -= modeled_change

	return (workState[0], score)


def createTestingShipWithCellsMissing(ship, n_cells_missing):
	alive = np.argwhere(ship == 1)

	removed_cells = []
	for _ in range(min(len(alive)-1, n_cells_missing)):
		cell_being_removed = random.randint(0, len(alive)-1)
		removed_cells.append(tuple(alive[cell_being_removed]))
		alive = np.delete(alive, cell_being_removed, axis=0)

	initialState = np.zeros_like(ship)

	initialState[alive[:, 0], alive[:, 1]] = 1
	initialState = initialState[None, :]
	initialState = torch.from_numpy(initialState)

	return initialState, removed_cells


def run_recursion_tests(pipeline, remove_counts_list, n_iters, n_items):
	with torch.no_grad():
		result_dict = {
			"n_iters" : [],
			"n_removed_cells" : [],
			"positive_scores" : [],
			"negative_scores" : [],
			"intactness_scores" : [],
			"score" : []
		}

		for n_removed_cells in remove_counts_list:
			for i in range(n_items):
				initialState, removed_cells = createTestingShipWithCellsMissing(random.choice(ships), n_removed_cells)

				# result = itercycle(pipeline, initialState, n_iters)[0].numpy() # remove extra batch dimention used by neural net
				result, score = itercycle(pipeline, initialState, n_iters)

				result_with_probs = [tuple(x) for x in list(np.argwhere(result.numpy() > 0))]
				original_alive_cells = [tuple(x) for x in list(np.argwhere(initialState[0].numpy() == 1))]

				positive_scores = [result[x[0], x[1]] for x in removed_cells]
				negative_scores = [result[x[0], x[1]] for x in result_with_probs if not x in original_alive_cells]
				intactness_list = [result[x[0], x[1]] for x in result_with_probs if x in original_alive_cells]

				# prevent mean on empty slice
				if not positive_scores:	positive_scores = [0]
				if not negative_scores:	negative_scores = [0]
				if not intactness_list:	intactness_list = [0]

				result_dict["n_iters"].append(n_iters)
				result_dict["n_removed_cells"].append(n_removed_cells)
				result_dict["positive_scores"].append(np.mean(positive_scores))
				result_dict["negative_scores"].append(np.mean(negative_scores))
				result_dict["intactness_scores"].append(np.mean(intactness_list))
				result_dict["score"].append(score)

		return pd.DataFrame(result_dict)


def displayResults(result):
	iter_results = result.groupby(["n_iters"]).aggregate(np.mean)[["positive_scores", "negative_scores", "intactness_scores", "score"]]
	removed_results = result.groupby(["n_removed_cells"]).aggregate(np.mean)[["positive_scores", "negative_scores", "intactness_scores", "score"]]
	
	plt.plot(iter_results["positive_scores"], label="positive")
	plt.plot(iter_results["negative_scores"], label="negative")
	plt.plot(iter_results["intactness_scores"], label="intactness")
	plt.plot(iter_results["score"], label="score")
	plt.xlabel("Number of iterations")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()

	plt.plot(removed_results["positive_scores"], label="positive")
	plt.plot(removed_results["negative_scores"], label="negative")
	plt.plot(removed_results["intactness_scores"], label="intactness")
	plt.plot(removed_results["score"], label="score")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()


def run_ship_network_tests():

	## LOADING MODELS
	pipeline = []
	pipe_names = ["5x5_included_no_score"]

	for item in pipe_names:
		model_path = os.path.join(ROOT_PATH, "models", item)
		model = ProbabilityFinder(1).double()
		model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
		model.eval()
		pipeline.append(model)

	MAX_ITER = 1	# the max amount of recursions
	MAX_REMOVE_COUNT = 40	# the maximum amount of cells to be removed

	test_n_iters = [i+1 for i in range(MAX_ITER)]
	remove_counts_list = [i+1 for i in range(MAX_REMOVE_COUNT)]
	test_n_spaceships = 25

	print(f"### N_SPACESHIPS = {test_n_spaceships} ###")
	n_iter_results = [run_recursion_tests(pipeline, remove_counts_list, n_iters, test_n_spaceships) for n_iters in test_n_iters]
	displayResults(pd.concat(n_iter_results))


def getMatrixScore(original_matrix, matrix):
	return torch.nn.MSELoss()(torch.from_numpy(original_matrix), torch.from_numpy(matrix)).item()


def runScoringTests(n_iters):

	model_name = "scoreOutputFile_1"
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
		for i, ship in enumerate(ships):
			n_cells_missing = random.randint(0, 100)
			testStructure, _ = createTestingShipWithCellsMissing(ship, n_cells_missing)
			score = model(testStructure).item()
			actualScore = getMatrixScore(testStructure.detach().numpy()[0], ship)

			scores["score"].append(score)
			scores["actualScore"].append(actualScore)
			scores["trainedShip"].append(i > 800)
			scores["n_cells_missing"].append(n_cells_missing)

	displayScoringTests(pd.DataFrame(scores))


def displayScoringTests(data):
	cellsMissing = data.groupby("n_cells_missing").aggregate(np.mean)
	trainedShip = data.groupby("trainedShip").aggregate(np.mean)
	print(trainedShip)

	plt.plot(cellsMissing["score"], label="Predicted Score")
	plt.plot(cellsMissing["actualScore"], label="Actual Score")
	plt.xlabel("Number of removed cells")
	plt.ylabel("Score")
	plt.legend(loc="upper left")
	plt.show()


if __name__ == "__main__":
	rle_reader = RleReader()
	filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)

	# run_ship_network_tests()
	runScoringTests(100)	# number input is number of iterations