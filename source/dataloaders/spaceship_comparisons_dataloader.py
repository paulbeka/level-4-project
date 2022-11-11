import sys

sys.path.insert(1, "C:\\Workspace\\level-4-project\\source")

from game_of_life import Game
from tools.rle_reader import RleReader
import numpy as np
import random
import os


class SpaceshipCompareDataloader:

	def __init__(self,
				n_samples,
				root_folder,
				delete_count,
				exclude_ratio = 0,
				ratio=0.8):
		
		self.n_samples = n_samples
		self.root_folder = root_folder
		self.n_delete_cells = delete_count
		self.exclude_ratio = exclude_ratio

		self.n_train_samples = int(n_samples * ratio)

		self.reader = RleReader()


	def loadSpaceships(self, spaceships, width, height):

		configurations = []

		for i in range(self.n_samples):
			spaceship = spaceships[i % len(spaceships)]

			newGrid = np.zeros((width, height))

			alive = np.argwhere(spaceship == 1)	
			# randomly remove parts of the spaceship (as you don't want to favour specific ships)
			if bool(random.getrandbits(1)):
				for i in range(min(self.n_delete_cells, len(alive) - 5)):	# make sure you don't delete all the cells
					alive = np.delete(alive, random.randint(0, len(alive)-1))

			newGrid[alive[:, 0], alive[:, 1]] = 1
			configurations.append(newGrid)

		return configurations


	def generateSpaceships(self, width, height):
		spaceships = self.reader.getFileArray(os.path.join(self.root_folder, "spaceships.txt"))

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

		train_dataset = spaceship_configs[0:self.n_train_samples]
		test_dataset = spaceship_configs[self.n_train_samples:]

		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

		game = Game(100, 100)
		game.renderItemList([item[0] for item in list(test_loader)])
		game.run()
		game.kill()

		return train_loader, test_loader


if __name__ == '__main__':
	dl = SpaceshipCompareDataloader(1000, os.path.join("C:\\Workspace\\level-4-project\\source\\data", "spaceship_identification"), 3)
	dl.loadData(100, 100)