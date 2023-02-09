import torch
import os, sys
import numpy as np
import random
import matplotlib.pyplot as plt

from networks.convolution_probability_network import ProbabilityFinder
from networks.score_predictor import ScoreFinder

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) # root of the source file
sys.path.insert(1, PROJECT_ROOT)

ROOT_PATH = os.path.abspath(os.getcwd())

from tools.rle_reader import RleReader  # in the tools class of source


THRESHOLD = 0.1  # value to consider a probability value as a cell


# for testing purposes
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


def transformToSolidStructure(matrix, threshold):
	alive = np.argwhere(matrix.detach().numpy() > threshold)
	newGrid = np.zeros_like(matrix.detach().numpy())
	newGrid[0, alive[:, 1], alive[:, 2]] = 1
	return torch.from_numpy(newGrid)


class Board:

	MAX_GRID = (10, 10)
	N_CONSIDERATIONS = 3

	def __init__(self, board):
		self.board = board
		self.visited = 0
		self.boardSize = self.board.shape[1:]


	def getPossibleActions(self):
		candidateStates = []
		
		probability_matrix = self.model(self.board)[0]
		candidate_cells = list(np.argwhere(probability_matrix.detach().numpy() != 0))
		candidate_cells.sort(key=lambda x: probability_matrix[x[0], x[1]])
		plt.imshow(probability_matrix.detach(), cmap='gray_r', interpolation='nearest')	
		plt.colorbar()
		plt.show()

		candidate_cells = candidate_cells[:Board.N_CONSIDERATIONS] + candidate_cells[-Board.N_CONSIDERATIONS:]
		print(candidate_cells)

		for candidate in candidate_cells:
			newGrid = self.board.clone()
			# print(candidate)
			newGrid[0, candidate[0], candidate[1]] = 1 - newGrid[0, candidate[0], candidate[1]] 

			candidateStates.append(Board(newGrid))

		return candidateStates


	def getScore(self):
		return self.scoringModel(self.board).item()


def nonStochasticProbabilityToStructure(probability_matrix):
	matrix = np.zeros_like(probability_matrix)
	alive = np.argwhere(probability_matrix > 0.5)
	matrix[alive[:, 0], alive[:, 1]] = 1
	return matrix


# implement this later
def stochasticProbabilityToStructure(probability_matrix):
	pass


def tree_search(max_depth, model, score_model, currentState):

	currentState = Board(currentState)
	bestState = currentState

	Board.model = model
	Board.scoringModel = score_model
	
	exploredStates = []
	actions = []

	for _ in range(max_depth):
		exploredStates.append(currentState)
		actions = [action for action in currentState.getPossibleActions() if not action in exploredStates]
		actions.sort(key=lambda x: x.getScore())
		
		if not actions:
			currentState = max(exploredStates, key=lambda x: x.getScore() - x.visited)
			currentState.visited += 1
		else:
			currentState = max(actions, key=lambda x: x.getScore())

		if currentState.getScore() < bestState.getScore():
			bestState = currentState
			# MAYBE ADD A SHIP TESTING METHOD

	print(f"Best score: {bestState.getScore()}")

	plt.imshow(bestState.board[0].detach(), cmap='gray_r', interpolation='nearest')	
	plt.colorbar()
	plt.show()

	return bestState
	

# LOAD THE PROBABILITY AND SCORING MODELS
MODEL_NAME = "5x5_included_20_pairs_epoch_1"
SCORE_MODEL_NAME = "deconstructScoreOutputFile_3"

model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

score_model_path = os.path.join(ROOT_PATH, "models", SCORE_MODEL_NAME)
score_model = ScoreFinder(1).double()
score_model.load_state_dict(torch.load(score_model_path))
score_model.eval()

# LOAD THE SHIPS FOR TESTING
rle_reader = RleReader()
filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
ships = rle_reader.getFileArray(filePath)

MAX_DEPTH = 2

# initialState = np.zeros((1, 10, 10))
initialState, removed_cells = createTestingShipWithCellsMissing(ships[55], 2)
print(f"Removed cells (y, x): {removed_cells}")
Board.MAX_GRID = (10, 10) 	# FIX THIS LATER

# plt.imshow(initialState[0], cmap='gray_r', interpolation='nearest')	
# plt.colorbar()
# plt.show()

tree_search(MAX_DEPTH, model, score_model, initialState)