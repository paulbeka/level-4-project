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


	def generateRandomRle(self):
		return np.random.randint(2, size=(self.box-random.randint(0, self.box-1), self.box-random.randint(0, self.box-1)))


if __name__ == "__main__":
	generator = RleGenerator()
	reader = RleReader("C:\\Workspace\\level-4-project\\source\\data\\30_30_all_spaceships.txt")

	rleList = reader.getFileArray()
	check = [generator.gridToRle(rle) for rle in rleList]
	actual_codes = reader.getRleCodes()

	for i, item in enumerate(actual_codes):
		# print(item)
		# print(check[i])
		print(check[i] == item)