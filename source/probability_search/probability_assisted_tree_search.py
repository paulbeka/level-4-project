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
from tools.testing import createTestingShipWithCellsMissing


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

		candidate_cells = candidate_cells[:Board.N_CONSIDERATIONS] + candidate_cells[-Board.N_CONSIDERATIONS:]

		for candidate in candidate_cells:
			newGrid = self.board.clone()
			newGrid[0, candidate[0], candidate[1]] = 1 - newGrid[0, candidate[0], candidate[1]] 

			candidateStates.append(Board(newGrid))

		return candidateStates


	def getScore(self):
		return self.scoringModel(self.board).item()


def tree_search(max_depth, model, score_model, currentState, number_of_returns=5):

	currentState = Board(currentState)
	bestStates = [currentState]
	bestScore = 1

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

		if currentState.getScore() < bestScore:
			bestScore = currentState.getScore()
			bestStates.append(currentState)

	return bestStates[len(bestStates)-number_of_returns:]


def strategicFill(inputGrid):
	# LOAD THE SHIPS FOR TESTING
	rle_reader = RleReader()
	filePath = os.path.join(ROOT_PATH, "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)
	testShip, removed = createTestingShipWithCellsMissing(ships[80], 1)
	print(removed)
	return testShip
	

# maybe could take the set of all results and find cells which are included in all of them
def optimizeInputGrid(inputGrid, results):
	return inputGrid


def search():

	# LOAD THE PROBABILITY AND SCORING MODELS
	MODEL_NAME = "5x5_included_20_pairs_epoch_4"
	SCORE_MODEL_NAME = "deconstructScoreOutputFile_1"

	model_path = os.path.join(ROOT_PATH, "models", MODEL_NAME)
	model = ProbabilityFinder(1).double()
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()

	score_model_path = os.path.join(ROOT_PATH, "models", SCORE_MODEL_NAME)
	score_model = ScoreFinder(1).double()
	score_model.load_state_dict(torch.load(score_model_path))
	score_model.eval()

	max_depth = 100
	ship_found = []

	size = (20, 20)
	inputGrid = np.zeros(size)
	inputGrid = strategicFill(inputGrid)

	n_iters = 10
	for i in range(n_iters):
		results = tree_search(max_depth, model, score_model, inputGrid)

		for result in results:
			data = outputShipData(np.array(result.board))
			if data:
				print(data["rle"])
				ship_found.append(data)
				
		inputGrid = optimizeInputGrid(inputGrid, results)
		print(f"Iteration {i+1}/{n_iters}")


if __name__ == "__main__":
	search()
