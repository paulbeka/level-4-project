# TODO: fix this class to identify ships with more than 2 dead cells between them

import numpy as np


class ObjectIdentifier:

	def __init__(self, maxSize, game):
		self.maxSize = maxSize
		self.game = game


	def identifyObjectProperties(self, config, maxPeriod):
		assert config.shape[0] < self.maxSize and config.shape[1] < self.maxSize, 
			f"Config must be smaller than max size of {self.maxSize}"
		board = np.zeros((self.maxSize, self.maxSize))
		board[config[:, 0]+(self.maxSize-config.shape[0])//2, config[:, 1]+(self.maxSize-config.shape[1])//2] = True

		period, pattern = self.evolveUntilPattern(board, maxPeriod)
		if period == -1 and pattern == None:
			return None


	def evolveUntilPattern(self, board, maxPeriod):
		currentConfig = board
		for i in range(maxPeriod):

			evolved_board = self.game.evolve(currentConfig)
			structures = self.findObjects(evolved_board)

			# check if pattern has died out
			if len(structures) == 0:
				return (-1, None)

			# this needs to be fixed to account for items that have a space of 2 or more between them
			currentConfig = structures[0]

			if np.array_equal(obj, currentConfig):
				usefulObjects.append(obj)
				break

