import numpy as np
from game_of_life import Game
import math


class BruteForceSearch:

	def __init__(self, width, height):
		self.width, self.height = width, height
		#self.game = Game(width, height)

	def search(self):

		configurations = []
		x_configs = []
		for row in range(2**self.width):
			val = bin(row).split("b")[1]
			val = "0" * (self.width-len(val)) + val
			x_configs.append(np.array([int(i) for i in list(val)]))

		curr = [0 for _ in range(self.height)]
		for i in range(1,self.height**len(x_configs)+1):
			configurations.append(np.array([x_configs[x] for x in curr]))
			if i % 100000 == 0:
				print(f"{(i / (self.height**len(x_configs)+1))*100}% done")
			for j in range(len(curr)):
				curr[j] = (i // len(x_configs)**j) % len(x_configs)


		print(np.array(configurations).shape)


	def checkNeighbours(self, config):
		cells = np.argwhere(self.cells == 1)



def main():
	newSearch = BruteForceSearch(4, 7)
	newSearch.search()


if __name__ == "__main__":
	main()

