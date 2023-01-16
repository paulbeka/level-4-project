import numpy as np
import sys
import torch

sys.path.insert(1, "C:\\Workspace\\level-4-project\\source")

from tools.rle_reader import RleReader


# n_pairs : the number of fake data for every spaceship config
def generateMockData(sizes, n_pairs):
	mockList = []
	for size in sizes:
		sizeMockList = [np.random.rand(size[0], size[1]) for _ in range(n_pairs)]
		mockList.append(sizeMockList)

	return mockList


# produce pairs of probability map -> solution to get to spaceship
def getPairSolutions(train_ratio, n_pairs, batch_size):
	rle_reader = RleReader()
	ships = rle_reader.getFileArray("C:\\Workspace\\level-4-project\\source\\data\\spaceship_identification\\spaceships_extended.txt")

	sizes = []
	for ship in ships:
		sizes.append(ship.shape)

	mock_data = generateMockData(sizes, n_pairs)
	data = []

	for ship, mock in zip(ships, mock_data):
		for mockItem in mock:
			solution = ship.copy() - mockItem
			data.append((mockItem, solution))

	n_train_samples = int(train_ratio * len(data))

	train_dataset = data[0:n_train_samples]
	test_dataset = data[n_train_samples:]

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader