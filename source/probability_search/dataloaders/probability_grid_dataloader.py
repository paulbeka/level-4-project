import numpy as np
import sys
import torch
import random
import os
import pickle


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, PROJECT_ROOT)

from tools.rle_reader import RleReader


# n_pairs : the number of fake data for every spaceship config
def generateMockData(sizes, n_pairs):
	mockList = []
	for size in sizes:
		sizeMockList = [np.random.rand(size[0], size[1]) for _ in range(n_pairs)]
		mockList.append(sizeMockList)

	return mockList


def getMatrixScore(original_matrix, matrix):
	return torch.nn.MSELoss()(torch.from_numpy(original_matrix), torch.from_numpy(matrix)).item()


# make a pair for each deconstructed ship part and the probability network assosiated
# MULTIPLE WAYS TO DO THIS:
# 1. Remove 1 random cell and train it
# 2. Remove 1 cell at a time and save every config	<--- WE'RE DOING THIS ONE RIGHT NOW
# 3. Remove one cell and make solution matrix that specific cell
def deconstructReconstructPairs(ships):
	data = []
	for ship in ships:
		ship_deconstructed = []
		alive = np.argwhere(ship == 1)
		for i in range(len(alive)-2):
			alive = np.delete(alive, random.randint(0, len(alive)-1), axis=0)
			tempGrid = np.zeros_like(ship)
			tempGrid[alive[:, 0], alive[:, 1]] = 1
			ship_deconstructed.append(tempGrid)
		data.append(ship_deconstructed)

	return data


# Remove n cells m different times in different locations
def ratioDeconstruct(ships, max_destruction_ratio, n_pairs, flip_other):
	data = []
	for location, ship in enumerate(ships):
		alive = np.argwhere(ship == 1)
		n_max_deconstruct = min(int(len(alive) * max_destruction_ratio), len(alive)-1)
		ship_deconstructed = []
		print(f"Ship {location}/{len(ships)} deconstructed.")
		for i in range(n_max_deconstruct):
			for _ in range(n_pairs):
				alive = np.delete(alive, [random.randint(0, len(alive)-1) for _ in range(i+1)], axis=0)
				tempGrid = np.zeros_like(ship)
				tempGrid[alive[:, 0], alive[:, 1]] = 1
				if flip_other:
					# flip an arbitrary number of dead cells
					n_dead_flips = random.randint(0, i+n_max_deconstruct)
					dead = np.argwhere(tempGrid == 0)
					dead_to_flip = dead[[random.randint(0, len(dead)-1) for _ in range(n_dead_flips)]]
					tempGrid[dead_to_flip[:, 0], dead_to_flip[:, 1]] = 1
				ship_deconstructed.append(tempGrid)
				alive = np.argwhere(ship == 1)

		data.append(ship_deconstructed)

	print("Ship deconstruction complete.")
	return data


# produce pairs of probability map -> solution to get to spaceship
def getPairSolutions(train_ratio, n_pairs, batch_size, data_type):
	rle_reader = RleReader()
	filePath = os.path.join(PROJECT_ROOT, "data", "spaceship_identification", "spaceships_extended.txt")
	ships = rle_reader.getFileArray(filePath)[:800] # IMPORTANT: LAST 180 ARE FOR TESTING PURPOSES

	sizes = []
	for ship in ships:
		sizes.append(ship.shape)

	# select the type of mock data
	# empty : only zeros
	# full : only 1s
	# random : random data  [TODO] add a density range 
	if data_type == "random":
		mock_data = generateMockData(sizes, n_pairs)
	elif data_type == "full":
		mock_data = [[np.ones(size)] for size in sizes]
		n_pairs = 1
	elif data_type == "empty":
		mock_data = [[np.zeros(size)] for size in sizes]
		n_pairs = 1
	elif data_type == "deconstruct":
		mock_data = deconstructReconstructPairs(ships)
	elif data_type == "advanced_deconstruct":
		mock_data = ratioDeconstruct(ships, 1, 5, True)
	else:
		raise Exception("Not a valid data training type: use random, full, or empty.")

	data = []
	for i, (ship, mock) in enumerate(zip(ships, mock_data)):
		for mockItem in mock:
			solution = ship.copy() - mockItem
			score = getMatrixScore(solution, mockItem)
			solution = solution.flatten()
			solution = np.append(score, solution)
			data.append((mockItem, solution))
		print(f"Mock item {i}/{len(ships)} finished.")

	n_train_samples = int(train_ratio * len(data))

	train_dataset = data[0:n_train_samples]
	test_dataset = data[n_train_samples:]

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader


def loadPairsFromFile(filename):
	data = np.load(filename, allow_pickle=True)
	with open('mock_data.pkl','rb') as f:
		x = pickle.load(f)

	train_loader = torch.utils.data.DataLoader(dataset=x, batch_size=1, shuffle=True)

	return train_loader, None