import numpy as np
from .game_of_life import Game
from .rle_reader import RleReader


# return if it is a ship, the speed, period, etc...
def outputShipData(pattern):
	searchSize = 128
	patternSize = pattern.shape[1:]
	expandedSizes = (patternSize[0] + searchSize, patternSize[1] + searchSize) # make space for spaceship movement
	originalRle = normalizedPatternToRle(patternIdentity(pattern[0]))

	game = Game(*expandedSizes)
	rle_reader = RleReader()

	boxedConfig = rle_reader.placeConfigInBox(pattern[0], *expandedSizes)
	game.cells = boxedConfig

	for i in range(searchSize):
		game.cells = game.getNextState()
		newRle = normalizedPatternToRle(patternIdentity(game.cells))
		
		if newRle == "":	# no spaceship- all cells died
			return {}

		if newRle == originalRle:
			newAlive = np.argwhere(game.cells == 1)
			originalAlive = np.argwhere(pattern[0] == 1)
			distance_x = abs(max(originalAlive[:, 0]) - max(newAlive[:, 0]) + searchSize //2)
			distance_y = abs(max(originalAlive[:, 1]) - max(newAlive[:, 1]) + searchSize //2)
			distance = (distance_x, distance_y)

			if distance:
				return {
					"period" : i,
					"distance_moved" : distance,
					"pattern" : pattern[0],
					"rle" : originalRle
				}

	return {}


def gridToRle(grid):
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
def normalizedPatternToRle(pattern):
	if len(pattern) < 1:
		return ""
	dims = [max(pattern[:, 0])+1, max(pattern[:, 1])+1]
	grid = np.zeros((dims[0], dims[1]))
	grid[pattern[:, 0], pattern[:, 1]] = True
	return gridToRle(grid)


# find the pattern identity of an object
def patternIdentity(pattern):
	modifiedPattern = pattern.copy()
	modifiedPattern = np.argwhere(pattern == 1)
	if len(modifiedPattern) < 1:
		return modifiedPattern
	reference = (min(modifiedPattern[:,0]), min(modifiedPattern[:,1]))
	modifiedPattern[:, 0] -= reference[0]
	modifiedPattern[:, 1] -= reference[1]
	return modifiedPattern