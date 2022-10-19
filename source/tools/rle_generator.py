import numpy as np
import random


class RleGenerator:

	def __init__(self, box=10):
		self.box = box


	def gridToRle(self, grid):
		rle = ""
		rowCount = 1
		for row in range(grid.shape[0]):
			currCount = 1
			currValue = grid[row, 0]
			for col in range(1, grid.shape[1]):
				if grid[row, col] != currValue:
					if currCount > 1:
						rle += str(currCount)

					if currValue:
						rle += 'b'
					else:
						rle += 'o'
					currCount = 0

				else:
					currCount += 1

			if currValue:
				if currCount > 1:
					rle += str(currCount)
				rle += 'o'

			# check if it was an empty row
			if currCount == grid.shape[1] and not currValue:
				rowCount += 1
				continue

			# add the $ count if the row size was above 1
			if rowCount > 1:
				rle += str(rowCount)
				rowCount = 1

			rle += "$"

		rle += "!"

		return rle


	def generateRandomRle(self):
		return np.random.randint(2, size=(self.box-random.randint(0, self.box-1), self.box-random.randint(0, self.box-1)))
