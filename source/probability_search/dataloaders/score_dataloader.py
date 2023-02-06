import numpy as np
import torch
import sys, os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, PROJECT_ROOT)

from tools.rle_reader import RleReader


def getMatrixScore(original_matrix, matrix):
	return torch.nn.MSELoss()(torch.from_numpy(original_matrix), torch.from_numpy(matrix)).item()


def createTestingShipWithCellsMissing(ship, n_cells_missing):
	alive = np.argwhere(ship == 1)

	removed_cells = []
	for _ in range(min(len(alive)-1, n_cells_missing)):
		cell_being_removed = random.randint(0, len(alive)-1)
		removed_cells.append(tuple(alive[cell_being_removed]))
		alive = np.delete(alive, cell_being_removed, axis=0)

	initialState = np.zeros_like(ship)

	initialState[alive[:, 0], alive[:, 1]] = 1
	initialState = initialState[None, :]

	return initialState[0], removed_cells


def ratioDeconstructWithAddedRandomCells(ships, max_destruction_ratio, n_pairs):
	data = []
	for location, ship in enumerate(ships):
		alive = np.argwhere(ship == 1)
		n_max_deconstruct = min(int(len(alive) * max_destruction_ratio), len(alive)-1)
		print(f"Ship {location}/{len(ships)} deconstructed.")
		for i in range(n_max_deconstruct):
			for _ in range(n_pairs):
				alive = np.delete(alive, [random.randint(0, len(alive)-1) for _ in range(i+1)], axis=0)
				tempGrid = np.zeros_like(ship)
				tempGrid[alive[:, 0], alive[:, 1]] = 1
				# flip an arbitrary number of dead cells
				n_dead_flips = random.randint(0, i+n_max_deconstruct)
				dead = np.argwhere(tempGrid == 0)
				dead_to_flip = dead[[random.randint(0, len(dead)-1) for _ in range(n_dead_flips)]]
				tempGrid[dead_to_flip[:, 0], dead_to_flip[:, 1]] = 1
				data.append((tempGrid, getMatrixScore(ship, tempGrid)))
				alive = np.argwhere(ship == 1)

	print("Ship deconstruction complete.")

	return data


def scoreDataloader(n_pairs, mode="ratio_deconstruct"):
	rle_reader = RleReader()
	filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)[:800] # IMPORTANT: LAST 180 ARE FOR TESTING PURPOSES
	
	sizes = []
	for ship in ships:
		sizes.append(ship.shape)

	data = []
	# for ship in ships:
	# 	if mode == "random":
	# 		mockList = [np.random.rand(ship.shape[0], ship.shape[1]) for _ in range(n_pairs)]
	# 		scores = [(mockItem, getMatrixScore(ship, mockItem)) for mockItem in mockList]
	# 	elif mode == "deconstruct":
	# 		mockList = [createTestingShipWithCellsMissing(ship, random.randint(0, 100))[0] for _ in range(n_pairs)]
	# 		scores = [(mockItem, getMatrixScore(ship, mockItem)) for mockItem in mockList]	
	# 	elif mode == "ratio_deconstruct":
	#		scores = ratioDeconstructWithAddedRandomCells(ships, 0.1, 1)

	data = ratioDeconstructWithAddedRandomCells(ships, 1, 10)

	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=True)

	return train_loader
