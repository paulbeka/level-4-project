from networks.lifenet_cnn import LifeNetCNN
from game_of_life import Game

import torch
import numpy as np

# now create an A* / tree search to find the correct cells to place
# first test to see if a complete spaceship will have a higher score than 
# a cell which won't create a spaceship

class PlaceSearcher:

	def __init__(self, width, height, path):
		self.width, self.height = width, height

		self.model = LifeNetCNN(2, 1).double()
		self.model.load_state_dict(torch.load(path))
		self.model.eval()


	def search(self, n_iters):
		searchBlock = np.random.randint(2, size=(self.width, self.height)).astype(np.double) # random config
		padWidth, padHeight = 100, 100
		#searchBlock = np.zeros((padWidth, padHeight)).astype(np.double)

		# TEST CONFIGURATION
		#searchBlock[0, 1] = 1.0
		#searchBlock[1, 0] = 1.0
		#searchBlock[0, 2] = 1.0
		#searchBlock[1, 2] = 1.0
		#searchBlock[2, 2] = 1.0

		searchBlock.resize(1, padWidth, padHeight)
		searchBlock = torch.from_numpy(searchBlock)

		n_flips = 100

		bestConfigs = []

		with torch.no_grad():
			for epoch in range(n_iters):
				print(f"Epoch {epoch+1}/{n_iters}")
				for _ in range(n_flips):

					possibleFlips = []
					threshold = 0.005

					# current state: the value we want flipped is not working

					first = torch.softmax(self.model(searchBlock), dim=1)

					for i in range(self.width):
						for j in range(self.height):
							searchBlock[0, i, j] = 1.0 - searchBlock[0, i, j]  # reverse the cell assignment
							second = torch.softmax(self.model(searchBlock), dim=1)
							searchBlock[0, i, j] = 1.0 - searchBlock[0, i, j]

							diff = second[0,1]-first[0,1]
							if diff > threshold:
								possibleFlips.append((i, j, diff.item()))
								
					if possibleFlips:
						possibleFlips.sort(key=lambda x: x[2], reverse=True)
						x, y, val = possibleFlips[0]
						searchBlock[0, x, y] = 1.0 - searchBlock[0, x, y]
						bestConfigs.append((searchBlock, torch.softmax(self.model(searchBlock), dim=1)[0, 1]))


					#print(f"Current score: {first[0,1]}")

		bestConfigs.sort(key=lambda x: x[1])
		print([item[1].item() for item in bestConfigs])
		return bestConfigs[-1][0]


	def checkSpaceship(self, config):
		# move forward x amount of times
		# check if repeat pattern and moved
		# return if not
		
		n_steps = 4
		for _ in range(n_steps):
			nextConfig = evolve(config)
			if nextConfig == config and min(nextConfig) < min(config):
				return True

		return False


	def displayResult(self, pattern):
		game = Game(self.width, self.height)
		game.renderItemList([(np.array(pattern[0]), 0)])
		game.run()
		game.kill()



if __name__ == '__main__':
	placeSearcher = PlaceSearcher(15, 15, "C:\\Workspace\\level-4-project\\source\\data\\models\\almost_identifier")
	result = placeSearcher.search(1)
	placeSearcher.displayResult(result)