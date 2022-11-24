import numpy as np
import random
from .rle_reader import RleReader


class RleGenerator:

	def __init__(self, width=10, height=10):
		self.width, self.height = width, height


	# Turns a given grid of 1s and 0s into an RLE code
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


	# Turns a normalizes pattern into RLE code
	def normalizedPatternToRle(self, pattern):
		dims = [max(pattern[:, 0])+1, max(pattern[:, 1])+1]
		grid = np.zeros((dims[0], dims[1]))
		grid[pattern[:, 0], pattern[:, 1]] = True
		return self.gridToRle(grid)


	# GENERATES RANDOM GRID, WITH OPTION OF RANDOM BOX SIZES
	# density        : represents the density of the cell counts
	# random_size_f  : flag to tell if the cells generated are put inside a random box
	def generateRandomGrid(self, density, random_size_f):
		try:
			num_decimals = str(density)[::-1].find('.')
			if random_size_f:
				grid = np.random.randint(10**num_decimals, size=(self.width-random.randint(0, self.width-1), self.height-random.randint(0, self.height-1)))
			else:
				grid = np.random.randint(10**num_decimals, size=(self.width, self.height))

			finalGrid = np.zeros((self.width, self.height))
			above_1_locations = np.argwhere(grid > density*(10**num_decimals))
			finalGrid[above_1_locations[:,0], above_1_locations[:,1]] = 1

			return finalGrid
		except:
			return self.generateRandomRle(density, random_size_f)


	# Takes in a list of box sizes and returns a list of random configs inside those boxes
	# n 		: number of grids to generate
	# density	: represents the density of the cell counts
	# box_sizes	: list of box sizes the random config can take
	def generateRandomInsideBoxSize(self, n, density, box_sizes):
		num_decimals = str(density)[::-1].find('.')
		randomGrids = []
		for i in range(n):
			shape = box_sizes[i % len(box_sizes)]
			grid = np.random.randint(10**num_decimals, size=shape)

			above_1_locations = np.argwhere(grid > density*(10**num_decimals))
			
			# Move the random box somewhere random on the grid
			# above_1_locations[:, 0] += random.randint(0, self.width-shape[0]-1)
			# above_1_locations[:, 1] += random.randint(0, self.height-shape[1]-1)

			newGrid = np.zeros((shape[0], shape[1]))
			newGrid[above_1_locations[:, 0], above_1_locations[:, 1]] = 1
			randomGrids.append(newGrid)

		return randomGrids


	# Returns a random RLE code
	# density        : represents the density of the cell counts
	# random_size_f  : flag to tell if the cells generated are put inside a random box
	def generateRandomRle(self, density, random_size_f):
		return self.gridToRle(self.generateRandomGrid(density, random_size_f))


	def generateRandomRleFile(self, count, density=0.5, random_size_f=False):
		string = ""
		for i in range(count):
			string += self.generateRandomRle(density, random_size_f) + "\n"

		with open("output.txt", "w") as f:
			f.write(string)

		print(f"{count} RLE objects created and written to output.txt")


if __name__ == "__main__":
	generator = RleGenerator(100, 100)

	# generating a random test file:
	generator.generateRandomRleFile(1000, density=0.5, random_size_f=True)