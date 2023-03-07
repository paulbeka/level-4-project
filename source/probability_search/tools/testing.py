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


# Remove n cells m different times in different locations
def mockRatioDeconstruct(ship, n_cells_missing, n_cells_extra):

	alive = np.argwhere(ship == 1)
	indexes_to_delete = [random.randint(0, len(alive)-1) for _ in range(n_cells_missing+1)]
	cells_missing = alive[indexes_to_delete]
	alive = np.delete(alive, indexes_to_delete, axis=0)

	tempGrid = np.zeros_like(ship)
	tempGrid[alive[:, 0], alive[:, 1]] = 1
	
	dead = np.argwhere(tempGrid == 0)
	dead_to_flip = dead[[random.randint(0, len(dead)-1) for _ in range(n_cells_extra)]]
	tempGrid[dead_to_flip[:, 0], dead_to_flip[:, 1]] = 1

	# returns the new object, the cells which were deleted, and the cells flipped which are not part of the ship
	return tempGrid, cells_missing, dead_to_flip