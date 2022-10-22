import numpy as np
from game_of_life import Game
from tools.object_identifier import ObjectIdentifier
import math


class BruteForceSearch:

	def __init__(self, width, height):
		self.width, self.height = width, height
		self.game = Game(width, height)
		# make sure the width set is higher than actual width
		self.objectIdentifier = ObjectIdentifier(width+2, self.game)

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
			objects = self.objectIdentifier.findObjects(configs[:, :, i])
			if objects:
				usefulObjects = self.objectIdentifier.findUsefulObjects(objects, 3)
				if usefulObjects:
					interestingObjects += usefulObjects

		# turn the objects into a board configuration
		interestingConfigs = [np.zeros((self.width, self.height)) for _ in range(len(interestingObjects))]
		for i, item in enumerate(interestingObjects):
			interestingConfigs[i][item[:, 0], item[:, 1]] = True

		# display objects
		self.game.renderItemList(interestingConfigs)
		self.game.run()


def main():
	newSearch = BruteForceSearch(3, 3)
	newSearch.search()


if __name__ == "__main__":
	main()

