import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt

from networks.convolution_probability_network import ProbabilityFinder
from networks.score_predictor import ScoreFinder

ROOT_PATH = os.path.abspath(os.getcwd()) # current root of the probability search

from tools.rle_reader import RleReader 
from tools.gol_tools import outputShipData, normalizedPatternToRle, patternIdentity
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
		return self.candidateAndScoringMethod()


	def candidateAndScoringMethod(self):
		candidateStates = []
		# MIGHT NEED TO CHANGE THESE VALUES IN ORDER TO ELIMINATE A GOOD AMOUNT OF CELLS- TO ALLOW TREE SEARCH TO RUN
		# MAYBE COULD MAKE THEM DYNAMIC: AS THE SEARCH CONTINUES, MAKE THEM MORE EXTREME TO ENCOURAGE SEARCH
		# OF OTHER BRANCHES IN THE SEARCHED LIST [CHECK THIS]
		positive_threshold = 0.3 #+ (self.level * 0.1)
		negative_threshold = -0.8# + (self.level * 0.1)
		
		probability_matrix = self.model(self.board)[0]
		candidate_cells = list(np.argwhere(abs(probability_matrix.detach().numpy()) > positive_threshold))

		candidates = []
		positive_additions = [tuple(candidate) for candidate in candidate_cells if probability_matrix[candidate[0], candidate[1]] > positive_threshold]
		negative_additions = [tuple(candidate) for candidate in candidate_cells if probability_matrix[candidate[0], candidate[1]] < negative_threshold]

		for i in range(self.N_CONSIDERATIONS):
			
			if len(positive_additions):
				max_item = max(positive_additions, key=lambda x: abs(probability_matrix[x[0], x[1]]))
				candidates.append(max(positive_additions, key=lambda x: abs(probability_matrix[x[0], x[1]])))
				positive_additions.remove(max_item)

			if len(negative_additions):
				max_item = max(negative_additions, key=lambda x: abs(probability_matrix[x[0], x[1]]))
				candidates.append(max_item)
				negative_additions.remove(max_item)

		for candidate in candidates:
			newGrid = self.board.clone()
			if probability_matrix[candidate[0], candidate[1]] > 0:
				newGrid[0, candidate[0], candidate[1]] = 1
			else:
				newGrid[0, candidate[0], candidate[1]] = 0

			candidateStates.append(Board(newGrid, self.level + 1))

		return candidateStates

	# MAYBE CHANGE THIS SCORING FUNCTION AND VISITED MODIFIER
	def getScore(self):
		return self.scoringModel(self.board).item() - self.visited


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
		actions = [action for action in currentState.getPossibleActions() if not action in exploredStates]

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
	MODEL_NAME = "probability_change_network"
	SCORE_MODEL_NAME = "scoring_network"

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

	all_results = []

	if testing_data:
		originalRle = normalizedPatternToRle(patternIdentity(np.array(testing_data["ship"])))

	for i in range(n_iters):

		# results = tree_search(max_depth, model, score_model, inputGrid)
		# max_depth as a function of iterations:
		results = tree_search(int(max_depth * (1 + (i*0.1))), model, score_model, inputGrid)

		all_results += results
		for result in results:
			if testing_data:
				if normalizedPatternToRle(patternIdentity(np.array(result.board[0]))) == originalRle:
					# print(originalRle)
					return all_results
				continue

			data = outputShipData(np.array(result.board))
			if data:
				print("FOUND SHIP: ")
				print(data["rle"])
				ship_found.append(data)
				return all_results
				
		inputGrid = optimizeInputGrid(inputGrid, results)
		print(f"Iteration {i+1}/{n_iters}")

	print(f"Search over. Found {len(ship_found)} spaceships.")
	return all_results

if __name__ == "__main__":
	found_structures = search()
