from tools.rle_reader import RleReader
import numpy as np
import random


class SpaceshipCompareDataloader:

	def __init__(self,
				n_samples,
				root_folder,
				ratio=0.8):
		
		self.n_samples = n_samples
		self.root_folder = root_folder

		self.n_train_samples = int(n_samples * ratio)

		self.reader = RleReader()


	def loadSpaceships(self):
		spaceships = self.reader.getFileArray(os.path.join(self.root_folder, "spaceships.txt"))

		modified_spaceships = []

		for i in range(self.n_samples):
			spaceship = spaceships[i % len(spaceships)]

			# randomly remove parts of the spaceship
			if bool(random.getrandbits(1)):