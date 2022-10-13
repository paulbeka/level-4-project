import numpy as np
from game_of_life import Game


class DepthSearch:

	def __init__(self, width, height):
		self.width, self.height = width, height
		self.game = Game(width, height)
		
		# the ship's search parameters
		self.search_speed = 3
		self.search_period = 4


	def search(self):
		x, y = 0, self.height//2
		self.recur(x, y)


	def recur(selfs, x, y):
		if y >= self.height-1:
			if x >= self.width-1:
				return
			self.recur(x+1, 0)

		# first check concequences of last choice

		# if wrong then back up

		# then if it's alright, chose new choice
		choice = np.random.choice(a=[False, True])


	def findStructure(self):
		pass


	def findGlider(self):
		pass



def main():
	ds = DepthSearch(40, 6)
	ds.search()


if __name == "__main__":
	main()