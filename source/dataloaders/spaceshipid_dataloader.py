# This is a dataloader class to load data that can identify spaceships
# Can be used to get random RLEs and random spaceship placements

import sys

sys.path.insert(1, "C:\\Workspace\\level-4-project\\source")

from tools.rle_reader import RleReader
from tools.rle_generator import RleGenerator
from game_of_life import Game
import random
import os
import torch
import numpy as np


class SpaceshipIdentifierDataLoader:

	def __init__(self, 
				n_samples,
				random_density,
				fixed_box_size,
				root_folder,
				batch_size=5,
				include_random_in_spaceship = False,
				ratio = 0.8):

		self.n_samples = n_samples
		self.random_density = random_density
		self.fixed_box_size = fixed_box_size
		self.root_folder = root_folder
		self.batch_size = batch_size
		self.include_random_in_spaceship = include_random_in_spaceship

		self.n_train_samples = int(n_samples * ratio)

		self.reader = RleReader()


	def loadSpaceships(self, width, height):
		spaceships = self.reader.getFileArray(os.path.join(self.root_folder, "spaceships.txt"))
		randomly_placed_spaceships = []

		generator = RleGenerator(width, height)
		game = Game(width, height, show_display=False)

		for i in range(self.n_samples):
			
			# need to loop around the only spaceships that we have available
			currSpaceship = spaceships[i % len(spaceships)]
			aliveLoc = np.argwhere(currSpaceship == 1)

			# using the max function to make sure it does not crash if shape[x]-1 = -1
			addWidthIndex = random.randint(0, max(width-currSpaceship.shape[0]-4, 0))
			addHeightIndex = random.randint(0, max(height-currSpaceship.shape[1]-4, 0))

			# add a bunch of noise around the spaceship by generating a random sample and placing ship there
			if self.include_random_in_spaceship:

				random_sample = generator.generateRandomGrid(self.random_density, False)
				x_min, y_min = min(aliveLoc[:,0])-2, min(aliveLoc[:,1])-2
				x_max, y_max = max(aliveLoc[:,0])+2, max(aliveLoc[:,1])+2

				temp_grid = np.zeros((x_max-x_min+1, y_max-y_min+1))
				temp_grid[np.copy(aliveLoc[:,0])-x_min, np.copy(aliveLoc[:,1])-y_min] = 1
				
				# find where 0s should be placed
				alive = np.argwhere(temp_grid == 1)
				for cell in alive:
					neighbours = Game.DOUBLE_NEIGHBOUR_TEMPLATE + cell
					temp_grid[neighbours[:, 0], neighbours[:, 1]] = 2

				# make sure the insides of the ship have space inside them
				for i, row in enumerate(temp_grid):
					cell_indexes = np.where(row == 2)
					if cell_indexes[0].any():
						temp_grid[i, cell_indexes[0][0]:cell_indexes[0][-1]] = 2

				place_zeros = np.argwhere(temp_grid == 2)
				random_sample[place_zeros[:, 0]+addWidthIndex, place_zeros[:, 1]+addHeightIndex] = 0  # make space for spaceship
				random_sample[aliveLoc[:, 0]+addWidthIndex+2, aliveLoc[:, 1]+addHeightIndex+2] = 1  # place spaceship (add 2 for pre-space-padding)

				randomly_placed_spaceships.append(random_sample)

			else:
				# remake the original generator
				aliveLoc[:, 0] += addWidthIndex
				aliveLoc[:, 1] += addHeightIndex

				grid = np.zeros((width, height))

				grid[aliveLoc[:,0], aliveLoc[:,1]] = 1
				randomly_placed_spaceships.append(grid)

		#game.renderItemList([(item, 1) for item in randomly_placed_spaceships])
		#game.run()
		game.kill()

		return [(item, 1) for item in randomly_placed_spaceships]


	# function used to generate an equal number of ships and random patterns in an n*m grid
	# width, height : dimentions of the grid
	def loadData(self, width, height):
		random_configs = [(item, 0) for item in self.reader.getFileArray(os.path.join(self.root_folder,"random_rles.txt"))]
		spaceship_configs = self.loadSpaceships(width, height)

		train_dataset = random_configs[0:self.n_train_samples] + spaceship_configs[0:self.n_train_samples]
		test_dataset = random_configs[self.n_train_samples:] + spaceship_configs[self.n_train_samples:]

		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
		return train_loader, test_loader


	def generateNewRandomRleDataset(self):
		generator = RleGenerator()



if __name__ == '__main__':
	dataloader_params = {'dataloader': [
										1000, 
										0.5, 
										False, 
										os.path.join("C:\\Workspace\\level-4-project\\source\\data", "spaceship_identification"), 
										5, 
										True
									],
					'width' : 100,
					'height': 100,
					'num_epochs' : 1,
					'batch_size' : 5
					}

	dataloader = SpaceshipIdentifierDataLoader(*dataloader_params['dataloader'])

	train_loader, test_loader = dataloader.loadData(100, 100)
