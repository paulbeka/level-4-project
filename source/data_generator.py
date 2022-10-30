from tools.rle_reader import RleReader
import random


class SpaceshipIdentifierDataGenerator:

	def __init__(self, 
				n_samples,
				random_density,
				fixed_box_size,
				root_folder):

		self.n_samples = n_samples
		self.random_density = random_density
		self.fixed_box_size = fixed_box_size
		self.root_folder = root_folder

		self.reader = RleReader()


	def loadSpaceships(self):
		spaceships = self.reader.getFileArray(self.root_folder + "\\spaceships.txt")
		for spaceship in spaceships:
			pass


	# function used to generate an equal number of ships and random patterns in an n*m grid
	# width, height : dimentions of the grid
	def loadSpaceshipIdentifierDataset(self, width, height):
		random_configs = self.reader.getFileArray(self.root_folder + "\\random_rles.txt")
		spaceship_configs = self.loadSpaceships()


if __name__ == '__main__':
	pass