import numpy as np
import torch
import os
import random
from rle_reader import RleReader
from gol_tools import outputShipData

ROOT_PATH = os.path.abspath(os.pardir) # current root of the probability search


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


def createTestingShipWithCellsAdded(ship, n_cells_added):
	dead = np.argwhere(ship == 0)

	added_cells = []
	for _ in range(min(len(dead)-1, n_cells_added)):
		cell_being_added = random.randint(0, len(dead)-1)
		added_cells.append(tuple(dead[cell_being_added]))
		dead = np.delete(dead, cell_being_added, axis=0)

	initialState = np.ones_like(ship)

	initialState[dead[:, 0], dead[:, 1]] = 0
	initialState = initialState[None, :]
	initialState = torch.from_numpy(initialState)

	return initialState, added_cells


def locationDifferencesBetweenTwoMatrixies(original, comparison):
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
	tempGrid = tempGrid[None, :]	# add extra batch dimention MIGHT BE WORTH IMPLMENTING THAT ONLY WHEN NEEDED

	# returns the new object, the cells which were deleted, and the cells flipped which are not part of the ship
	return torch.from_numpy(tempGrid), cells_missing, dead_to_flip


def checkSpaceshipFinderAlgorithm():
	rle_reader = RleReader()
	filePath = os.path.join(ROOT_PATH, "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)
	for i, ship in enumerate(ships):
		ship = ship[None, :]

		if outputShipData(ship):
			print(f"Ship {i} : OK.")
		else:
			print(f"[ERROR] Ship {i} failed.")


if __name__ == "__main__":
	checkSpaceshipFinderAlgorithm()