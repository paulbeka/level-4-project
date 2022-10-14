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
			if objects:
				usefulObjects = self.findUsefulObjects(objects)
				if usefulObjects:
					interestingObjects.append(usefulObjects)

		print(interestingObjects)


	def findObjects(self, config):

		alive = np.argwhere(config == 1)
		#print(alive)
		objects = []

		while len(alive):
			cell = alive[0]
			structure = [cell]
			alive = alive[1:]

			neighbours = self.game.getValidNeighbours(cell)

			neighbor_stack = neighbours[np.argwhere(config[neighbours[:, 0], neighbours[:, 1]] == 1)].reshape(-1, 2)

			while len(neighbor_stack):
				np.append(neighbor_stack, np.argwhere(config[self.game.getValidNeighbours(neighbor_stack[0])]))
				if config[neighbor_stack[0][0], neighbor_stack[0][1]] == 1:
					structure.append(neighbor_stack[0])
					print(np.where(alive == neighbor_stack[0]))

				neighbor_stack = neighbor_stack[1:, :]

			if len(structure) > 1:
				objects.append(np.array(structure))

		#print(objects)
		return objects


	def findUsefulObjects(self, objects):
		usefulObjects = []

		for obj in objects:
			currentConfig = obj
			for i in range(1, self.search_period):

				evolved_board = self.game.evolve(currentConfig)
				structures = self.findObjects(evolved_board)
				# print(structures)

				if len(structures) > 1 or len(structures) == 0:
					break
				currentConfig = structures[0]

				if currentConfig[0].flatten() == obj.flatten():
					usefulObjects.append(obj)
					break

def main():
	newSearch = BruteForceSearch(2,2)
	newSearch.search()


if __name__ == "__main__":
	main()

