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
			if i % 5000 == 0:
				print(f"{100*i/configs.shape[2]:.2f}% done")
			objects = self.findObjects(configs[:, :, i])
			if objects:
				usefulObjects = self.findUsefulObjects(objects)
				if usefulObjects:
					interestingObjects += usefulObjects

		# turn the objects into a board configuration
		interestingConfigs = [np.zeros((self.width, self.height)) for _ in range(len(interestingObjects))]
		for i, item in enumerate(interestingObjects):
			interestingConfigs[i][item[:, 0], item[:, 1]] = True

		# display objects
		self.game.renderItemList(interestingConfigs)
		self.game.run()


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


	def findUsefulObjects(self, objects):
		usefulObjects = []

		for obj in objects:
			currentConfig = obj
			for i in range(1, self.search_period):
				evolved_board = self.game.evolve(currentConfig)
				structures = self.findObjects(evolved_board)

				if len(structures) == 0:
					break
				currentConfig = structures[0]

				if np.array_equal(obj, currentConfig):
					usefulObjects.append(obj)
					break

		return usefulObjects


def main():
	newSearch = BruteForceSearch(4, 4)
	newSearch.search()


if __name__ == "__main__":
	main()

