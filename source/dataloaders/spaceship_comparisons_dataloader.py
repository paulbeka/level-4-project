import sys

sys.path.insert(1, "C:\\Workspace\\level-4-project\\source")

from game_of_life import Game
from tools.rle_reader import RleReader
import numpy as np
import random
import os
import torch


class SpaceshipCompareDataloader:

	def __init__(self,
				n_samples,
				root_folder,
				delete_coef,
				exclude_ratio = 0,
				batch_size=1,
				test_to_train_ratio=0.8):
		
		self.n_samples = n_samples
		self.root_folder = root_folder
		self.delete_coef = delete_coef
		self.exclude_ratio = exclude_ratio
		self.batch_size = batch_size	

		self.n_train_samples = int(n_samples * test_to_train_ratio)

		self.reader = RleReader()


	# Generate half real spaceships, half messed up spaceships
	def loadSpaceships(self, spaceships, width, height):

		configurations = []

		for i in range(self.n_samples):
			spaceship = spaceships[i % len(spaceships)]
			alive = np.argwhere(spaceship == 1)	

			w = int(max(alive[:, 0]) - min(alive[:, 0])) + 1
			h = int(max(alive[:, 1]) - min(alive[:, 1])) + 1

			widthBoundary, heightBoundary = 100, 100

			newGrid = np.zeros((widthBoundary, heightBoundary))

			# randomly remove parts of the spaceship (as you don't want to favour specific ships)
			fakeShip = random.randint(0, 1)
			if fakeShip:
				for i in range(int(self.delete_coef * len(alive))+1):	# delete cells
					alive = np.delete(alive, random.randint(0, len(alive)-1), axis=0)

			newGrid[alive[:, 0], alive[:, 1]] = 1
			configurations.append((newGrid, 1 - fakeShip))  # fakeShip is boolean check for the dataset

		return configurations


	# load the dataset with the correct exclusion ratio
	def generateSpaceships(self, width, height):
		spaceships = self.reader.getFileArray(os.path.join(self.root_folder, "spaceships_extended.txt"))

		if self.exclude_ratio > 0:
			excluded_spaceship_indexes = random.sample(range(0, len(spaceships)-1), len(spaceships)*self.exclude_spaceships_ratio)
			excluded_spaceships = [spaceships[i] for i in excluded_spaceship_indexes]
			non_excluded_spaceships = [spaceships[i] for i in range(len(spaceships)-1) if i not in excluded_spaceship_indexes]

			excluded_configs = self.loadSpaceships(excluded_spaceships, width, height)[0:self.n_train_samples]
			non_excluded_configs = self.loadSpaceships(non_excluded_spaceships, width, height)[self.n_train_samples:]

			return excluded_configs + non_excluded_configs

		else:
			return self.loadSpaceships(spaceships, width, height)


	def loadData(self, width, height):
		spaceship_configs = self.generateSpaceships(width, height)
		random.shuffle(spaceship_configs)  # make sure the ships are not ordered

		train_dataset = spaceship_configs[0:self.n_train_samples]
		test_dataset = spaceship_configs[self.n_train_samples:]

		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

		game = Game(100, 100)
		game.renderItemList([item for item in test_dataset])
		game.run()
		game.kill()

		return train_loader, test_loader


if __name__ == '__main__':
	dl = SpaceshipCompareDataloader(1000, os.path.join("C:\\Workspace\\level-4-project\\source\\data", "spaceship_identification"), 3)
	dl.loadData(100, 100)
