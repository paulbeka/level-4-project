# TODO: fix this class to identify ships with more than 2 dead cells between them

import numpy as np
from .rle_generator import RleGenerator


class ObjectIdentifier:

	def __init__(self, maxSize, game):
		self.maxSize = maxSize
		self.game = game
		self.rleGenerator = RleGenerator()


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

			# this is a queue of alive cells to which valid neighbours are added, to form a structure
			# it terminates when a structure has 2 dead cells between its cells and other structures
			neighbor_q = list(self.game.getValidNeighbours(cell).reshape(-1, 2))
			while neighbor_q:
				currentSearch = neighbor_q.pop(0)
				if not any((currentSearch == x).all() for x in structure):
					if config[currentSearch[0], currentSearch[1]] == 1:
						# check that the cell is still in the alive list
						if not any((currentSearch == x).all() for x in alive):
							continue
						neighbor_q += list(self.game.getValidNeighbours(currentSearch).reshape(-1, 2))
						structure.append(currentSearch)
						alive = [x for x in alive if not np.array_equal(x, currentSearch)]
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
				# this is temporary - needs to return full list of objects
				usefulObjects.append(properties[0])

		return usefulObjects


	### IDENTIFYING OBJECT PROPERTIES ###
	# expand this method later to include more data analysis (that's why it exists)
	def identifyObjectProperties(self, config, maxPeriod):
		assert max(config[:,0]) < self.maxSize and max(config[:,1]) < self.maxSize, f"Config must be smaller than max size of {self.maxSize}"

		pattern, period, speed = self.evolveUntilPattern(config, maxPeriod)

		# check item found has properties
		if period == -1:
			return None

		else:
			return (pattern, period, speed)


	def evolveUntilPattern(self, config, maxPeriod):
		currentConfig = config + self.maxSize//2 # put structure in the board center
		normalizedPattern = self.game.patternIdentity(currentConfig)

		# make sure the pattern has a padding of maxSize to the highest width in the pattern
		dimention = max(max(config[:,0]), max(config[:,1])) + self.maxSize
		# make the new board to be evolved
		board = np.zeros((dimention, dimention))
		board[currentConfig[:, 0], currentConfig[:, 1]] = True

		for i in range(1, maxPeriod):

			board = self.game.evolve(board)
			structures = self.findObjects(board)

			# check if pattern has died out- stop execution
			if len(structures) == 0:
				return (None, -1, -1)

			# this needs to be fixed to account for items that have a space of 2 or more between them
			for structure in structures:
				if np.array_equal(self.game.patternIdentity(structure), normalizedPattern):
					distance = abs(np.argmin(structure) - np.argmin(currentConfig))
					return (self.rleGenerator.normalizedPatternToRle(normalizedPattern), i, distance)

		return (None, -1, -1)



# Testing
if __name__ == '__main__':
	pass