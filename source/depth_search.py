import numpy as np
from game_of_life import Game


class DepthSearch:

	def __init__(self, width, height):
		self.width, self.height = width, height
		self.game = Game(width, height)
		self.search_space = np.full((width, height), 2)
		
		# the ship's search parameters
		# self.search_speed = 3
		# self.search_period = 4


	def search(self):
		pass


	def findStructure(self):
		pass


	def findGlider(self):
		pass



def main():
	ds = DepthSearch(10, 10)
	ds.search()


if __name == "__main__":
	main()