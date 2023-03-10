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
	N_CONSIDERATIONS = 3

	def __init__(self, board, level):
		self.board = board
		self.visited = 0
		self.level = level
		self.boardSize = self.board.shape[1:]

	def __eq__(self, other):
		return np.array_equal(self.board, other.board)


	def getPossibleActions(self):
		return self.candidateAndScoringMethod(iteration)


	def candidateAndScoringMethod(self):
		candidateStates = []
		# MIGHT NEED TO CHANGE THESE VALUES IN ORDER TO ELIMINATE A GOOD AMOUNT OF CELLS- TO ALLOW TREE SEARCH TO RUN
		# MAYBE COULD MAKE THEM DYNAMIC: AS THE SEARCH CONTINUES, MAKE THEM MORE EXTREME TO ENCOURAGE SEARCH
		# OF OTHER BRANCHES IN THE SEARCHED LIST [CHECK THIS]
		positive_threshold = 0.15 + (self.level * 0.1)
		negative_threshold = -0.6 + (self.level * 0.1)
		
		probability_matrix = self.model(self.board)[0]
		candidate_cells = list(np.argwhere(probability_matrix.detach().numpy() != 0))

		candidates = []
		for i in range(self.N_CONSIDERATIONS):
			positive_additions = [candidate for candidate in candidate_cells if probability_matrix[candidate[0], candidate[1]] > positive_threshold]
			negative_additions = [candidate for candidate in candidate_cells if probability_matrix[candidate[0], candidate[1]] < negative_threshold]

			if len(positive_additions):
				candidates.append(max(positive_additions, key=lambda x: abs(probability_matrix[x[0], x[1]])))
			if len(negative_additions):
				candidates.append(max(negative_additions, key=lambda x: abs(probability_matrix[x[0], x[1]])))

		for candidate in candidates:
			newGrid = self.board.clone()
			if probability_matrix[candidate[0], candidate[1]] > 0:
				newGrid[0, candidate[0], candidate[1]] = 1
			else:
				newGrid[0, candidate[0], candidate[1]] = 0

			candidateStates.append(Board(newGrid, self.level + 1))

		return candidateStates


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

	threshold = 0.18 
	newMatrix = np.zeros_like(result)
	aboveHalf = np.argwhere(result > threshold)
	newMatrix[aboveHalf[:, 0], aboveHalf[:, 1]] = 1

	return torch.from_numpy(result)


def tree_search(max_depth, model, score_model, currentState, number_of_returns=10):

	currentState = Board(currentState, 0)
	bestStates = [currentState]
	bestScore = 1

	Board.model = model
	Board.scoringModel = score_model
	
	exploredStates = []
	actions = []

	for i in range(max_depth):
		exploredStates.append(currentState)
		actions = [action for action in currentState.getPossibleActions(i) if not action in exploredStates]

		if not actions:
			currentState = max(exploredStates, key=lambda x: x.getScore())
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
	SCORE_MODEL_NAME = "brandnewScoringNetworkEpoch_10"

	model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
	model = ProbabilityFinder(1).double()
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()

	score_model_path = os.path.join(ROOT_PATH, "models", SCORE_MODEL_NAME)
	score_model = ScoreFinder(1).double()
	score_model.load_state_dict(torch.load(score_model_path, map_location=torch.device('cpu')))
	score_model.eval()

	ship_found = []

	inputGrid = initialInput  # use the user supplied input
	if isinstance(inputGrid, type(None)):
		inputGrid, removed = strategicFill(inputGrid) #no user supplied input- strategic fill

	if testing_data:
		inputGrid = testing_data["initialInput"] # use input from testing data

	all_results = []

	# IDEA: CHANGE THE MAX DEPTH AS A FUNCTION OF NUMBER OF ITERATIONS
	for i in range(n_iters):
		if testing_data:
			# is this feature necessary?
			results = tree_search(max_depth, model, score_model, inputGrid, debug_list=testing_data["removed"], debug_ship=testing_data["og_ship"])
		else:
			# results = tree_search(max_depth, model, score_model, inputGrid)
			# max_depth as a function of iterations:
			results = tree_search(int(max_depth * (1 + (i*0.1))), model, score_model, inputGrid)


		all_results += results
		for result in results:
			data = outputShipData(np.array(result.board))
			if data:
				print(data["rle"])
				ship_found.append(data)
				return all_results
				
		inputGrid = optimizeInputGrid(inputGrid, results)
		print(f"Iteration {i+1}/{n_iters}")

	return all_results

if __name__ == "__main__":
	search()
