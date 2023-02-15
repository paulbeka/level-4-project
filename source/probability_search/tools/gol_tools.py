import numpy as np
from .game_of_life import Game
from .rle_reader import RleReader


# return if it is a ship, the speed, period, etc...
def outputShipData(pattern):
	searchSize = 32

	patternSize = pattern.shape[1:]
	expandedSizes = (patternSize[0] + searchSize, patternSize[1] + searchSize) # make space for spaceship movement
	game = Game(*expandedSizes)
	rle_reader = RleReader()

	boxedConfig = rle_reader.placeConfigInBox(pattern[0], *expandedSizes)
	game.cells = boxedConfig

	for i in range(searchSize):
		game.evolve()



def x():
	pass