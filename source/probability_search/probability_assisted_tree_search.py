import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from networks.convolution_probability_network import ProbabilityFinder


ROOT_PATH = os.path.abspath(os.getcwd())


# turn a flattened output grid into a grid
def modelOutputToGridAndScore(input_shape, model_output):
	flattened_grid = model_output[:model_output.shape[0]-1]
	grid = flattened_grid.reshape(input_shape)
	score = model_output[-1]
	return (grid, score)


class Board:

	MAX_GRID = (10, 10)
	N_CONSIDERATIONS = 2

	def __init__(self, board, x, y):
		self.board = board
		self.visited = 0
		self.x, self.y = x, y


	def getPossibleActions(self):
		states = []
		
		probability_matrix = self.model(self.board)
		candidate_cells = list(np.argwhere(probability_matrix != 0))
		candidate_cells.sort(key=lambda x: probability_matrix[x[0], x[1]])

		candidate_cells = candidate_cells[:Board.N_CONSIDERATIONS] + candidate_cells[-Board.N_CONSIDERATIONS:]

		for candidate in candidate_cells:
			newGrid = self.board.copy()
			newGrid[candidate[0], candidate[1]] += probability_matrix[candidate[0], candidate[1]]


		return states


def nonStochasticProbabilityToStructure(probability_matrix):
	matrix = np.zeros_like(probability_matrix)
	alive = np.argwhere(probability_matrix > 0.5)
	matrix[alive[:, 0], alive[:, 1]] = 1
	return matrix


# implement this later
def stochasticProbabilityToStructure(probability_matrix):
	pass


def tree_search(max_depth, model, currentState):

	bestState = currentState

	Board.model = model
	
	exploredStates = []
	actions = []

	for _ in range(max_depth):
		exploredStates.append(currentState)
		actions = [action for action in currentState.getPossibleActions() if not action in exploredStates]

		if not actions:
			currentState = max(exploredStates, key=lambda x: x.getScore() - x.visited)
			currentState.visited += 1
		else:
			currentState = max(actions, key=lambda x: x.getScore())

		if currentState.getScore() > bestState.getScore():
			bestState = currentState

	print(bestState.getScore())
	

MAX_DEPTH = 100
MODEL_NAME = "3_epoch_rand_addition_advanced_deconstruct"

model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
model = ProbabilityFinder(1).double()
model.load_state_dict(torch.load(model_path))
model.eval()

initialState = np.zeros((10, 10))
Board.MAX_GRID = (10, 10) 	# FIX THIS LATER

tree_search(MAX_DEPTH, model, initialState)
