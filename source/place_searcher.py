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


	def search(self, n_iters):
		#searchBlock = np.random.randint(2, size=(self.width, self.height)).astype(np.double)
		searchBlock = np.zeros((self.width, self.height)).astype(np.double)
		searchBlock.resize(1, 100, 100)
		searchBlock = torch.from_numpy(searchBlock)
		with torch.no_grad():
			for epoch in range(n_iters):
				print(f"Epoch {epoch+1}/{n_iters}")
				for i in range(self.width):
					for j in range(self.height):
						first = self.model(searchBlock)
						searchBlock[0, i, j] = float(not bool(int(searchBlock[0, i,j])))  # reverse the cell assignment
						second = self.model(searchBlock)
						if first[0, 1] >= second[0, 1]:	# check if the cell is better on or off
							searchBlock[0, i, j] = float(not bool(int(searchBlock[0, i,j])))

				print(f"Current score: {first[0,1]}")


		return searchBlock


	def displayResult(self, pattern):
		game = Game(self.width, self.height)
		game.renderItemList([np.array(pattern)])
		game.run()
		game.kill()



if __name__ == '__main__':
	placeSearcher = PlaceSearcher(15, 15, "C:\\Workspace\\level-4-project\\source\\data\\models\\noise_or_spaceship")
	result = placeSearcher.search(100)
	placeSearcher.displayResult(result)