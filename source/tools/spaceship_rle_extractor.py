# This file is meant to extract the RLE patterns of a spaceship
# for all configurations + directions it can take

from game_of_life import Game
from tools.rle_reader import RleReader
from tools.rle_generator import RleGenerator

import numpy as np	


def extract_all_configs(FILENAME):

	game = Game()
	reader = RleReader()
	generator = RleGenerator()

	rle_list = reader.getRleCodes(FILENAME)
	final_rle_list = []

	for rle in rle_list:
		config_list = []

		config = reader.getConfig(rle)
		currConfig = game.evolve(config.copy())

		while config != currConfig:
			for _ in range(4):
				# spin around 4 times to get all the configurations
				config_list.append(game.patternIdentity(currConfig))
				np.rot90(config)

			currConfig = game.evolve(currConfig)

		for config in config_list:
			final_rle_list.append(generator.normalizedPatternToRle(config))

	return final_rle_list



def main():

	FILENAME = ""
	SAVE_FILE = ""

	extended_rle_list = extract_all_configs(FILENAME)

	with open(SAVE_FILE, "w") as f:
		f.writelines(extended_rle_list)


if __name__ == "__main__":
	main()