# This is a dataloader class to load data that can identify spaceships
# Can be used to get random RLEs and random spaceship placements

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
			grid = np.zeros((width, height))
			
			# need to loop around the only spaceships that we have available
			currSpaceship = spaceships[i % len(spaceships)]
			aliveLoc = np.argwhere(currSpaceship == 1)

			# using the max function to make sure it does not crash if shape[x]-1 = -1
			addWidthIndex = random.randint(0, max(width-currSpaceship.shape[0]-1, 0))
			addHeightIndex = random.randint(0, max(height-currSpaceship.shape[1]-1, 0))

			aliveLoc[:, 0] += addWidthIndex
			aliveLoc[:, 1] += addHeightIndex

			grid[aliveLoc[:,0], aliveLoc[:,1]] = 1

			# add a bunch of noise around the spaceship by generating a random sample and placing ship there
			if self.include_random_in_spaceship:

				random_sample = generator.generateRandomGrid(self.random_density, False)
				x_min, y_min = min(aliveLoc[:,0]), min(aliveLoc[:,1])
				x_max, y_max = max(aliveLoc[:,0]), max(aliveLoc[:,1])

				temp_grid = np.zeros((x_max-x_min+1, y_max-y_min+1))
				print(temp_grid)
				temp_grid[aliveLoc[:,0]-x_min, aliveLoc[:,1]-y_min] = 1

				# find the min/max of every row/col and set all inside the range to 0
				for i, row in enumerate(temp_grid):
					alive = np.argwhere(row == 1)
					if not len(alive):
						temp_grid[i, :] = 2
						continue
					min_x, max_x = min(alive)[0], max(alive)[0]
					temp_grid[i, 0:min_x-1] = 2
					temp_grid[i, max_x+1:] = 2

				for i, col in enumerate(temp_grid.T):
					alive = np.argwhere(col == 1)
					if not len(alive):
						temp_grid[:, i] = 2
						continue
					min_y, max_y = min(alive)[0], max(alive)[0]
					temp_grid[0:min_y, i] += 2
					temp_grid[max_y:, i] += 2

				# place zeros and spaceship
				zero_placements = np.argwhere(1 < temp_grid < 4)
				random_sample[zero_placements[:,0]+x_min, zero_placements[:,1]+y_min] = 0
				random_sample[aliveLoc[:,0], aliveLoc[:,1]] = 1

				randomly_placed_spaceships.append(random_sample)

			else:
				randomly_placed_spaceships.append(grid)


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
	pass