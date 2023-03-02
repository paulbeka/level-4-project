import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from networks.convolution_probability_network import ProbabilityFinder
from networks.score_predictor import ScoreFinder

ROOT_PATH = os.path.abspath(os.getcwd()) # current root of the probability search

from tools.rle_reader import RleReader 
from tools.gol_tools import outputShipData
from tools.testing import createTestingShipWithCellsMissing, locationDifferencesBetweenTwoMatrixies


class Board:

	MAX_GRID = (10, 10)
	N_CONSIDERATIONS = 1

	def __init__(self, board):
		self.board = board
		self.visited = 0
		self.boardSize = self.board.shape[1:]

	def __eq__(self, other):
		return np.array_equal(self.board, other.board)


	def getPossibleActions(self, debug_list=[], debug_ship=None):
		return self.candidateAndScoringMethod()


	def candidateAndScoringMethod(self):
		candidateStates = []
		
		probability_matrix = self.model(self.board)[0]
		candidate_cells = list(np.argwhere(probability_matrix.detach().numpy() != 0))
		candidate_cells.sort(key=lambda x: probability_matrix[x[0], x[1]])

		candidate_cells = candidate_cells[:Board.N_CONSIDERATIONS] + candidate_cells[-Board.N_CONSIDERATIONS:]
		for candidate in candidate_cells:
			newGrid = self.board.clone()
			newGrid[0, candidate[0], candidate[1]] = 1 - newGrid[0, candidate[0], candidate[1]] 

			candidateStates.append(Board(newGrid))

		return candidateStates


	# this is shit - fix it maybe?
	def greedyMethod(self):
		probability_matrix = self.model(self.board)[0]
		candidate_cells = list(np.argwhere(probability_matrix.detach().numpy() != 0))
		candidate = max(candidate_cells, key=lambda x: abs(probability_matrix[x[0], x[1]].item()))
		newGrid = self.board.clone()
		newGrid[0, candidate[0], candidate[1]] = 1 - newGrid[0, candidate[0], candidate[1]]
		return [Board(newGrid)]


	def iterativeChangeMethod(self):
		newGrid = modelChangeIteratively(self.model, self.board.clone(), 1)
		return [Board(newGrid)]

	def getScore(self):
		return self.scoringModel(self.board).item()


# returns a model change matrix
def modelChangeIteratively(model, initialState, n_iters):
	workState = initialState.detach()
	for _ in range(n_iters):
		modeled_change = model(workState).detach()
		result = (workState + modeled_change).numpy()[0]
		remove_ones = np.argwhere(result > 1)
		result[remove_ones[:, 0], remove_ones[:, 1]] = 1
		remove_zeros = np.argwhere(result < 0)
		result[remove_zeros[:, 0], remove_zeros[:, 1]] = 0
		result = result[None, :]
		workState = torch.from_numpy(result)

	threshold = np.mean(result) # normalize to find the threshold
	aboveHalf = np.argwhere(result > threshold)
	result[aboveHalf[:, 0], aboveHalf[:, 1]] = 1
	belowHalf = np.argwhere(result <= threshold)
	result[belowHalf[:, 0], belowHalf[:, 1]] = 0

	return torch.from_numpy(result)


def tree_search(max_depth, model, score_model, currentState, number_of_returns=5, debug_list=[], debug_ship=None):

	currentState = Board(currentState)
	bestStates = [currentState]
	bestScore = 1

	Board.model = model
	Board.scoringModel = score_model
	
	exploredStates = []
	actions = []

	for _ in range(max_depth):
		exploredStates.append(currentState)
		actions = [action for action in currentState.getPossibleActions(debug_list=debug_list, debug_ship=debug_ship) if not action in exploredStates]

		if not actions:
			currentState = max(exploredStates, key=lambda x: x.getScore() - x.visited)
			currentState.visited += 1
		else:
			currentState = max(actions, key=lambda x: x.getScore())

		if currentState.getScore() < bestScore:
			bestScore = currentState.getScore()
			bestStates.append(currentState)

	return bestStates[len(bestStates)-number_of_returns:]


def strategicFill(inputGrid):
	# LOAD THE SHIPS FOR TESTING
	rle_reader = RleReader()
	filePath = os.path.join(ROOT_PATH, "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)
	testShip, removed = createTestingShipWithCellsMissing(ships[80], 5)
	print(f"Removed: {removed}")
	return testShip, removed
	

# maybe could take the set of all results and find cells which are included in all of them
def optimizeInputGrid(inputGrid, results):
	commonCellsOnlyGrid = np.ones_like(inputGrid).astype(int)
	for grid in results:
		commonCellsOnlyGrid = commonCellsOnlyGrid & np.array(grid.board).astype(int)
	return torch.from_numpy(commonCellsOnlyGrid.astype(np.double))


def search(initialInput=None, n_iters=10, size=(20, 20), max_depth=100, testing_data={}):

	# LOAD THE PROBABILITY AND SCORING MODELS
	MODEL_NAME = "5x5_included_20_pairs_epoch_4"
	SCORE_MODEL_NAME = "deconstructScoreOutputFile_2"

	model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
	model = ProbabilityFinder(1).double()
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()

	score_model_path = os.path.join(ROOT_PATH, "models", SCORE_MODEL_NAME)
	score_model = ScoreFinder(1).double()
	score_model.load_state_dict(torch.load(score_model_path))
	score_model.eval()

	ship_found = []

	inputGrid = initialInput  # use the user supplied input
	if isinstance(inputGrid, type(None)):
		inputGrid, removed = strategicFill(inputGrid) #no user supplied input- strategic fill

	if testing_data:
		inputGrid = testing_data["initialInput"] # use input from testing data

	all_results = []

	for i in range(n_iters):
		if testing_data:
			# is this feature necessary?
			results = tree_search(max_depth, model, score_model, inputGrid, debug_list=testing_data["removed"], debug_ship=testing_data["og_ship"])
		else:
			results = tree_search(max_depth, model, score_model, inputGrid)

		for result in results:
			data = outputShipData(np.array(result.board))
			if data:
				print(data["rle"])
				ship_found.append(data)
				
		all_results += results
		inputGrid = optimizeInputGrid(inputGrid, results)
		print(f"Iteration {i+1}/{n_iters}")

	return all_results

if __name__ == "__main__":
	search()
