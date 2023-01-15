import numpy as np
import torch
import os

from networks.lifenet_cnn import LifeNetCNN
from game_of_life import Game


ROOT_PATH = os.path.abspath(os.getcwd())


class Board:

	MAX_GRID = (10, 10)

	def __init__(self, board, x, y):
		self.board = board
		self.visited = 0
		self.x, self.y = x, y

	def __eq__(self, other):
		return np.array_equal(self.board, other.board)


	def getPossibleActions(self):
		states = []
		if self.x >= Board.MAX_GRID[0]-1 or self.y >= Board.MAX_GRID[1]-1:
			return []

		if self.x < self.y:
			newBoard = self.board.copy()
			states.append(Board(newBoard, self.x + 1, self.y))	# one board puts cell = 0
			newBoard[self.x + 1, self.y] = 1
			states.append(Board(newBoard, self.x + 1, self.y))	# other puts cell = 1
		else:
			newBoard = self.board.copy()
			states.append(Board(newBoard, 0, self.y + 1))	# one board puts cell = 0
			newBoard[self.x + 1, self.y] = 1
			states.append(Board(newBoard, 0, self.y + 1))	# other puts cell = 1

		return states


	def getScore(self):
		padded = np.zeros((100, 100))
		alive = np.argwhere(self.board == 1)
		padded[alive[:, 0], alive[:, 1]] = 1
		return torch.softmax(Board.model(padded), dim=1)[0, 1].item()  # [1] is the chance it is a spaceship


def displayBoard(width, height, board):
	game = Game(width, height)
	game.renderItemList([(board, 0)])
	game.run()
	game.kill()

def tree_search(width, height):

	currentState = Board(np.zeros((width, height)), 0, 0)
	bestState = currentState

	model_path = os.path.join(ROOT_PATH, "models", "comparisons_extended")

	model = LifeNetCNN(2, 1).double()
	model.load_state_dict(torch.load(model_path))
	model.eval()

	Board.model = model
	Board.MAX_GRID = (10, 10)
	
	max_depth = 100
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
	displayBoard(width, height, bestState.board)


if __name__ == "__main__":
	tree_search(10, 10)
	