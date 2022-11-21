from networks.lifenet_cnn import LifeNetCNN
from game_of_life import Game

import torch
import numpy as np


class PlaceSearcher:

	def __init__(self, width, height, path):
		self.width, self.height = width, height

		self.model = LifeNetCNN(2, 1).double()
		self.model.load_state_dict(torch.load(path))
		self.model.eval()


	def search(self):
		searchBlock = np.random(0, 1, (self.width, self.height))
		for i in range(self.width):
			for j in range(self.height):
				first = self.model(searchBlock)
				searchBlock[i, j] = int(not bool(int(searchV)))  # reverse the cell assignment
				second = self.model(searchBlock)
				if first > second:	# check if the cell is better on or off
					searchBlock[i, j] = int(not bool(int(searchV)))

		return searchBlock


	def displayResult(self, pattern):
		game = Game()



if __name__ == '__main__':
	placeSearcher = PlaceSearcher(100, 100, "")
	placeSearcher.search()