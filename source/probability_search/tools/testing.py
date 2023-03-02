import numpy as np
import torch
import random


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
	initialState = torch.from_numpy(initialState)

	return initialState, removed_cells


def locationDifferencesBetweenTwoMatrixies(original, comparison):
	# print("############## ORIGINAL ##############")
	# print(original)
	# print("############## NEW ##############")
	# print(comparison)
	difference = original - comparison
	extra = [tuple(x) for x in np.argwhere(difference < 0)]
	missing = [tuple(x) for x in np.argwhere(difference > 0)]
	return extra, missing
