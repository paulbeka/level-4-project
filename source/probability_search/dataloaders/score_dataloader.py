import numpy as np
import torch
import os
import random

ROOT_PATH = os.path.abspath(os.getcwd()) # current root of the probability search

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
		print(f"PHASE 2/2: Ship {location}/{len(ships)}")
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


def scoreDataloader(n_pairs, mode="deconstruct"):
	rle_reader = RleReader()
	filePath = os.path.join(ROOT_PATH, "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)[:800] # IMPORTANT: LAST 180 ARE FOR TESTING PURPOSES
	
	sizes = []
	for ship in ships:
		sizes.append(ship.shape)

	data = []
	for i, ship in enumerate(ships):
		print(f"PHASE 1/2: Ship {i}/{len(ships)}")
		mockList = [np.random.rand(ship.shape[0], ship.shape[1]) for _ in range(n_pairs)]
		data += [(mockItem, getMatrixScore(ship, mockItem)) for mockItem in mockList]
		
		mockList = [createTestingShipWithCellsMissing(ship, random.randint(0, 100))[0] for _ in range(n_pairs)]
		data += [(mockItem, getMatrixScore(ship, mockItem)) for mockItem in mockList]	

	data += ratioDeconstructWithAddedRandomCells(ships, 1, 25)

	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=True)

	return train_loader
