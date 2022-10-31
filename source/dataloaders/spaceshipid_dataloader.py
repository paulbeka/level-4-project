# This is a dataloader class to load data that can identify spaceships
# Can be used to get random RLEs and random spaceship placements

from tools.rle_reader import RleReader
import random
import torch


class SpaceshipIdentifierDataLoader:

	def __init__(self, 
				n_samples,
				random_density,
				fixed_box_size,
				root_folder,
				batch_size=5,
				ratio = 0.8):

		self.n_samples = n_samples
		self.random_density = random_density
		self.fixed_box_size = fixed_box_size
		self.root_folder = root_folder
		self.batch_size = batch_size

		self.n_train_samples = int(n_samples * ratio)

		self.reader = RleReader()


	def loadSpaceships(self, width, height):
		spaceships = self.reader.getFileArray(self.root_folder + "\\spaceships.txt")
		randomly_placed_spaceships = []

		for spaceship in spaceships:
			if width-spaceship.shape[0]-1 > 0:
				spaceship[:, 0] += random.randint(0, width-spaceship.shape[0]-1)
			if height-spaceship.shape[1]-1 > 0:
				spaceship[:, 1] += random.randint(0, height-spaceship.shape[1]-1)

		return [(item, 1) for item in randomly_placed_spaceships]


	# function used to generate an equal number of ships and random patterns in an n*m grid
	# width, height : dimentions of the grid
	def loadSpaceshipIdentifierDataset(self, width, height):
		random_configs = [(item, 0) for item in self.reader.getFileArray(self.root_folder + "\\random_rles.txt")]
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