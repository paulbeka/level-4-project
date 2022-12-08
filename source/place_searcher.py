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
		padWidth, padHeight = 100, 100
		searchBlock = np.zeros((padWidth, padHeight)).astype(np.double)
		searchBlock[0, 1] = 1.0
		searchBlock[1, 0] = 1.0
		#searchBlock[0, 2] = 1.0
		searchBlock[1, 2] = 1.0
		searchBlock[2, 2] = 1.0
		searchBlock.resize(1, padWidth, padHeight)
		searchBlock = torch.from_numpy(searchBlock)

		n_flips = 10

		with torch.no_grad():
			for epoch in range(n_iters):
				print(f"Epoch {epoch+1}/{n_iters}")
				for _ in range(n_flips):

					possibleFlips = []
					threshold = 0.0000001

					# current state: the value we want flipped is not working

					for i in range(self.width):
						for j in range(self.height):
							first = torch.softmax(self.model(searchBlock), dim=1)
							searchBlock[0, i, j] = 1.0 - searchBlock[0, i, j]  # reverse the cell assignment
							second = torch.softmax(self.model(searchBlock), dim=1)
							searchBlock[0, i, j] = 1.0 - searchBlock[0, i, j]

							diff = second[0,0]-first[0,0]
							if diff > threshold:
								possibleFlips.append((i, j, diff.item()))
								
					possibleFlips.sort(key=lambda x: x[2], reverse=True)
					x, y, val = possibleFlips[0]
					searchBlock[0, x, y] = 1.0 - searchBlock[0, x, y]


					#print(f"Current score: {first[0,1]}")


		return searchBlock


	def displayResult(self, pattern):
		game = Game(self.width, self.height)
		game.renderItemList([(np.array(pattern[0]), 0)])
		game.run()
		game.kill()



if __name__ == '__main__':
	placeSearcher = PlaceSearcher(15, 15, "C:\\Workspace\\level-4-project\\source\\data\\models\\almost_identifier")
	result = placeSearcher.search(1)
	placeSearcher.displayResult(result)