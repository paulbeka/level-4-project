import numpy as np


class Board:

	def __init__(self, board):
		self.board = board
		self.visited = 0


	def getPossibleMoves(self):
		pass


def depth_first_search(width, height):

	currentSearch = np.zeros((width, height))
	
	max_depth = 100
	visit_stack = []

	for _ in range(max_depth):
		moves = currentSearch.getPossibleMoves()
		
