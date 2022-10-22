# TODO: fix this class to identify ships with more than 2 dead cells between them

import numpy as np


class ObjectIdentifier:

	def __init__(self, maxSize, game):
		self.maxSize = maxSize
		self.game = game


	# needs to be updated to find objects with more than 2 dead cells between them
	def findObjects(self, config):
		alive = list(np.argwhere(config == 1))
		objects = []
		# go through all the alive cells
		while alive:
			cell = alive[0]
			structure = [cell]
			alive = alive[1:]
			searched = []

			neighbor_q = list(self.game.getValidNeighbours(cell).reshape(-1, 2))

			while neighbor_q:
				currentSearch = neighbor_q.pop(0)
				if not any((currentSearch == x).all() for x in structure):
					if config[currentSearch[0], currentSearch[1]] == 1:
						neighbor_q += list(self.game.getValidNeighbours(currentSearch).reshape(-1, 2))
						structure.append(currentSearch)
						alive = [x for x in alive if not (x==currentSearch).all()]
					else:
						for n in list(self.game.getValidNeighbours(currentSearch).reshape(-1, 2)):
							if config[n[0], n[1]] == 1 and not any((n == x).all() for x in structure):
								neighbor_q.append(n)
				else:
					continue

			if len(structure) > 2:
				objects.append(np.array(list(structure)))

		return objects


	def findUsefulObjects(self, objects, maxPeriod):
		usefulObjects = []

		for item in objects:
			properties = self.identifyObjectProperties(item, maxPeriod)
			if properties != None:
				usefulObjects.append(properties)

		return usefulObjects


	### IDENTIFYING OBJECT PROPERTIES ###
	def identifyObjectProperties(self, config, maxPeriod):
		assert config.shape[0] < self.maxSize and config.shape[1] < self.maxSize, f"Config must be smaller than max size of {self.maxSize}"
		board = np.zeros((self.maxSize, self.maxSize))
		board[config[:, 0]+(self.maxSize-config.shape[0])//2, config[:, 1]+(self.maxSize-config.shape[1])//2] = True

		pattern, period, speed = self.evolveUntilPattern(board, maxPeriod)
		if pattern == None:
			return None

		else:
			return (pattern, period, speed)


	def evolveUntilPattern(self, board, maxPeriod):
		currentConfig = np.argwhere(board == 1)
		normalizedPattern = self.game.patternIdentity(currentConfig)
		for i in range(1, maxPeriod):
			evolved_board = self.game.evolve(currentConfig)
			structures = self.findObjects(evolved_board)

			# check if pattern has died out
			if len(structures) == 0:
				return (None, -1, -1)

			# this needs to be fixed to account for items that have a space of 2 or more between them
			currentConfig = structures[0]

			if np.array_equal(self.game.patternIdentity(currentConfig), normalizedPattern):
				distance = abs(np.argmin(currentConfig) - np.argmin(normalizedPattern))
				return (normalizedPattern, i, distance)

		return (None, -1, -1)



# Testing
if __name__ == '__main__':
	pass