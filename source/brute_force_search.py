import numpy as np
from game_of_life import Game
import math


class BruteForceSearch:

	def __init__(self, width, height):
		self.width, self.height = width, height
		#self.game = Game(width, height)

	def search(self):
		config = np.zeros((self.width, self.height))
		for row in range(2**self.width):
			val = bin(row).split("b")[1]
			val = "0" * (self.width-len(val)) + val
			print(val)



def main():
	newSearch = BruteForceSearch(5, 5)
	newSearch.search()


if __name__ == "__main__":
	main()

