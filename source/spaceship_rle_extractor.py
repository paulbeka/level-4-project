# This file is meant to extract the RLE patterns of a spaceship
# for all configurations + directions it can take

from game_of_life import Game
from tools.rle_reader import RleReader
from tools.rle_generator import RleGenerator

import numpy as np	


def extract_all_configs(FILENAME):

	game = Game(100, 100)
	reader = RleReader()
	generator = RleGenerator()

	rle_list = reader.getRleCodes(FILENAME)
	final_rle_list = []

	for rle in rle_list:
		config_list = []

		config = reader.placeConfigInBox(reader.getConfig(rle), 100, 100)
		currConfig = game.evolve(config.copy())

		while not np.array_equal(game.patternIdentity(config), game.patternIdentity(currConfig)):
			for _ in range(4):
				# spin around 4 times to get all the configurations
				config_list.append(game.patternIdentity(currConfig))
				currConfig = np.rot90(currConfig)

			currConfig = game.evolve(currConfig)

		for config in config_list:
			final_rle_list.append(generator.normalizedPatternToRle(config))

	return final_rle_list



def main():

	FILENAME = "C:\\Workspace\\level-4-project\\source\\data\\spaceship_identification\\spaceships.txt"
	SAVE_FILE = "C:\\Workspace\\level-4-project\\source\\data\\spaceship_identification\\spaceships_extended.txt"

	extended_rle_list = extract_all_configs(FILENAME)

	with open(SAVE_FILE, "w") as f:
		for line in extended_rle_list:
			f.write(line + "\n")


if __name__ == "__main__":
	main()