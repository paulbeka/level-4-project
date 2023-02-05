import numpy as np
import torch
import sys, os

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
	initialState = torch.from_numpy(initialState)

	return initialState, removed_cells

def scoreDataloader(n_pairs):
	rle_reader = RleReader()
	filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)[:800] # IMPORTANT: LAST 180 ARE FOR TESTING PURPOSES
	
	sizes = []
	for ship in ships:
		sizes.append(ship.shape)

	data = []
	for ship in ships:
		mockList = [np.random.rand(ship.shape[0], ship.shape[1]) for _ in range(n_pairs)]
		scores = [(ship, getMatrixScore(ship, mockItem)) for mockItem in mockList]
		data += scores

	train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=True)

	return train_loader
