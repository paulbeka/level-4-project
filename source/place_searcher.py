from networks.lifenet_cnn import LifeNetCNN
from game_of_life import Game
from tools.rle_reader import RleReader

import torch
import numpy as np
import random

# now create an A* / tree search to find the correct cells to place
# first test to see if a complete spaceship will have a higher score than 
# a cell which won't create a spaceship

class PlaceSearcher:

	def __init__(self, width, height, path):
		self.width, self.height = width, height

		self.model = LifeNetCNN(2, 1).double()
		self.model.load_state_dict(torch.load(path))
		self.model.eval()


	def testLoader(self):
		rle_reader = RleReader()
		arr = rle_reader.getFileArray("C:\\Workspace\\level-4-project\\source\\data\\spaceship_identification\\spaceships_extended.txt")
		selected_ship = arr[25]
		return selected_ship


	def search(self, n_iters):
		#searchBlock = np.random.randint(2, size=(self.width, self.height)).astype(np.double) # random config
		#searchBlock = np.zeros((padWidth, padHeight)).astype(np.double) # empty config
		alive = np.argwhere(self.testLoader() == 1)
		for _ in range(1):
			delete_cell = random.randint(0, len(alive)-1)
			print(alive[delete_cell])
			alive = np.delete(alive, delete_cell, axis=0)

		searchBlock = np.zeros((1, 100, 100))
		searchBlock[0, alive[:, 0], alive[:, 1]] = 1
		searchBlock = torch.from_numpy(searchBlock)

		n_flips = 1

		bestConfigs = []

		with torch.no_grad():
			for epoch in range(n_iters):
				print(f"Epoch {epoch+1}/{n_iters}")
				for _ in range(n_flips):

					possibleFlips = []
					threshold = 0

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
						print(possibleFlips)
						x, y, val = possibleFlips[0]
						searchBlock[0, x, y] = 1.0 - searchBlock[0, x, y]
						bestConfigs.append((searchBlock, torch.softmax(self.model(searchBlock), dim=1)[0, 1]))


		bestConfigs.sort(key=lambda x: x[1])
		print([item[1].item() for item in bestConfigs])
		if bestConfigs:
			return bestConfigs[-1][0]
		else:
			return searchBlock


 	### NEED TO DEVELOP PROPERLY ###
	def checkSpaceship(self, config):
		# move forward x amount of times
		# check if repeat pattern and moved
		# return if not
		
		n_steps = 4
		for _ in range(n_steps):
			nextConfig = evolve(config)
			if nextConfig == config and min(nextConfig) != min(config):
				return True

		return False


	def displayResult(self, pattern):
		game = Game(self.width, self.height)
		game.renderItemList([(np.array(pattern[0]), 0)])
		game.run()
		game.kill()



if __name__ == '__main__':
	placeSearcher = PlaceSearcher(15, 15, "C:\\Workspace\\level-4-project\\source\\data\\models\\test")
	result = placeSearcher.search(1)
	placeSearcher.displayResult(result)