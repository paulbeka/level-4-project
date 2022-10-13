import numpy as np
from game_of_life import Game
import math


class BruteForceSearch:

	def __init__(self, width, height):
		self.width, self.height = width, height
		self.game = Game(width, height)

		self.search_period = 4


	def search(self):

		interestingObjects = []

		# create all configurations for a NxM grid
		configs = np.zeros((self.width, self.height, 0))
		for row in range(2**(self.width*self.height)):
			val = bin(row).split("b")[1]
			val = "0" * ((self.width*self.height)-len(val)) + val
			configs = np.append(configs, np.array([int(i) for i in list(val)]).reshape(self.width, self.height, 1), axis=2)

		for i in range(configs.shape[2]):
			objects = self.findObjects(configs[:, :, i])
			usefulObjects = self.findUsefulObjects(objects)
			if usefulObjects:
				interestingObjects.append(usefulObjects)

		print(interestingObjects)


	def findObjects(self, config):

		alive = np.argwhere(config == 1)
		objects = []

		for i in range(len(alive)-1):
			cell = alive[i]
			structure = [cell]

			neighbours = self.game.getValidNeighbours(cell)

			alive_neighbors = neighbours[np.argwhere(config[neighbours[:, 0], neighbours[:, 1]] == 1)].reshape(-1, 2)

			while len(alive_neighbors):
				np.append(alive_neighbors, np.argwhere(config[self.game.getValidNeighbours(alive_neighbors[0])]))
				structure.append(alive_neighbors[0])
				alive_neighbors = alive_neighbors[1:, :]
			if len(structure) > 1:
				objects.append(np.array(structure))


		return objects



	def findUsefulObjects(self, objects):
		for obj in objects:
			for i in range(1, self.search_period):
				if self.game.evolve(obj, i) == obj:
					return True


def main():
	newSearch = BruteForceSearch(3, 3)
	newSearch.search()


if __name__ == "__main__":
	main()

