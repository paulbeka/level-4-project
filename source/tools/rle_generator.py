import numpy as np
import random
from rle_reader import RleReader


class RleGenerator:

	def __init__(self, box=10):
		self.box = box


	def gridToRle(self, grid):
		rle = f"x={grid.shape[0]},y={grid.shape[1]}\n"
		rowCount = 1
		for row in range(grid.shape[1]):

			currCount = 1
			currValue = grid[0, row]

			for col in range(1, grid.shape[0]):
				if grid[col, row] != currValue:
					# place the $ characters
					if rowCount > 1:
						rle = rle[:-1]
						rle += str(rowCount) + "$"
						rowCount = 1

				
					if currCount > 1:
						rle += str(currCount)

					if currValue:
						rle += 'o'
					else:
						rle += 'b'

					currValue = grid[col, row]
					currCount = 1

				else:
					currCount += 1

			if currValue:
				if currCount > 1:
					rle += str(currCount)
				rle += 'o'

			# check if it was an empty row
			if currCount >= grid.shape[0] and currValue == 0:
				rowCount += 1
				continue

			rle += "$"


		# remove unneccessary characters
		while rle[-1] != 'o':
			rle = rle[:-1]

		rle += "!"

		return rle


	def normalizedPatternToRle(self, pattern):
		dims = [max(pattern[:, 0])+1, max(pattern[:, 1])+1]
		grid = np.zeros((dims[0], dims[1]))
		grid[pattern[:, 0], pattern[:, 1]] = True
		return self.gridToRle(grid)


	# density        : represents the density of the cell counts
	# random_size_f  : flag to tell if the cells generated are put inside a random box
	def generateRandomRle(self, density, random_size_f):
		try:
			num_decimals = str(density)[::-1].find('.')
			if random_size_f:
				grid = np.random.randint(10*num_decimals, size=(self.box-random.randint(0, self.box-1), self.box-random.randint(0, self.box-1)))
			else:
				grid = np.random.randint(10*num_decimals, size=(self.box, self.box))

			finalGrid = np.zeros((self.box, self.box))
			above_1_locations = np.argwhere(grid > density*(10**num_decimals))
			finalGrid[above_1_locations[:,0], above_1_locations[:,1]] = 1
			return self.gridToRle(finalGrid)
		except:
			return self.generateRandomRle(density, random_size_f)


	def generateRandomRleFile(self, count, density=0.5, random_size_f=False):
		string = ""
		for i in range(count):
			string += self.generateRandomRle(density, random_size_f) + "\n"

		with open("output.txt", "w") as f:
			f.write(string)

		print(f"{count} RLE objects created and written to output.txt")


if __name__ == "__main__":
	generator = RleGenerator(box=100)
	print(generator.generateRandomRle(0.5, False))
	generator.generateRandomRleFile(1000, density=0.5, random_size_f=False)