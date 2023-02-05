import torch
import os, sys
import numpy as np
import random
import matplotlib.pyplot as plt

from networks.convolution_probability_network import ProbabilityFinder

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


# turn a flattened output grid into a grid
def modelOutputToGridAndScore(input_shape, model_output):
	# BAD FIX- FIX IT QUICK
	if len(model_output.shape) > 1:
		model_output = model_output[0]
	flattened_grid = model_output[:model_output.shape[0]-1]
	grid = flattened_grid.reshape(input_shape)
	score = model_output[-1]
	return (grid, score)


def transformToSolidStructure(matrix, threshold):
	alive = np.argwhere(matrix.detach().numpy() > threshold)
	newGrid = np.zeros_like(matrix.detach().numpy())
	newGrid[0, alive[:, 1], alive[:, 2]] = 1
	return torch.from_numpy(newGrid)


class Board:

	MAX_GRID = (10, 10)
	N_CONSIDERATIONS = 2

	def __init__(self, board, score):
		if len(board.shape) < 3:
			board = board[None, :]	# fix this
		self.board = board
		self.score = score
		self.visited = 0
		self.boardSize = self.board.shape[1:]


	def getPossibleActions(self):
		candidateStates = []
		
		probability_matrix, _ = modelOutputToGridAndScore(self.boardSize, self.model(self.board))

		candidate_cells = list(np.argwhere(probability_matrix.detach().numpy() != 0))
		candidate_cells.sort(key=lambda x: probability_matrix[x[0], x[1]])

		candidate_cells = candidate_cells[:Board.N_CONSIDERATIONS] + candidate_cells[-Board.N_CONSIDERATIONS:]
		print(candidate_cells)

		for candidate in candidate_cells:
			newGrid = self.board.clone()
			newGrid[0, candidate[0], candidate[1]] = 1 - newGrid[0, candidate[0], candidate[1]] 

			newGrid = transformToSolidStructure(newGrid, THRESHOLD)

			_, newScore = modelOutputToGridAndScore(self.boardSize, self.model(newGrid))
			candidateStates.append(Board(newGrid, newScore))

		return candidateStates


	def getScore(self):
		return self.score


def nonStochasticProbabilityToStructure(probability_matrix):
	matrix = np.zeros_like(probability_matrix)
	alive = np.argwhere(probability_matrix > 0.5)
	matrix[alive[:, 0], alive[:, 1]] = 1
	return matrix


# implement this later
def stochasticProbabilityToStructure(probability_matrix):
	pass


def tree_search(max_depth, model, currentState):

	_, currentScore = modelOutputToGridAndScore(currentState.shape[1:], model(currentState)[0])
	currentState = Board(currentState, currentScore)
	bestState = currentState
	Board.model = model
	
	exploredStates = []
	actions = []

	for _ in range(max_depth):
		exploredStates.append(currentState)
		actions = [action for action in currentState.getPossibleActions() if not action in exploredStates]
		actions.sort(key=lambda x: x.getScore())

		if not actions:
			currentState = max(exploredStates, key=lambda x: x.getScore() + x.visited)
			currentState.visited += 1
		else:
			currentState = max(actions, key=lambda x: x.getScore())

		if currentState.getScore() < bestState.getScore():
			bestState = currentState

	print(f"Best score: {bestState.getScore()}")

	plt.imshow(bestState.board[0].detach(), cmap='gray_r', interpolation='nearest')	
	plt.colorbar()
	plt.show()

	return bestState
	

MAX_DEPTH = 30
MODEL_NAME = "OUTPUT_SEND_THIS_BY_EMAIL"

model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

rle_reader = RleReader()
filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
ships = rle_reader.getFileArray(filePath)

# initialState = np.zeros((1, 10, 10))
initialState, removed_cells = createTestingShipWithCellsMissing(ships[39], 10)
print(f"Removed cells (y, x): {removed_cells}")
Board.MAX_GRID = (10, 10) 	# FIX THIS LATER

plt.imshow(initialState[0], cmap='gray_r', interpolation='nearest')	
plt.colorbar()
plt.show()

tree_search(MAX_DEPTH, model, initialState)