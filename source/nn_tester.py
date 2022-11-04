import torch
import os


class NeuralNetworkTester:

	def __init__(self, Network):
		self.model = Network()


	def loadNeuralNetwork(self, path):
		self.load_state_dict(torch.load(path))
		self.model.eval()


	def loadData(self, DataLoader, parameterPath):
		with open(parameterPath) as f:
			params = [item.rstrip() for item in f.readlines()]
		dataloader = DataLoader(*params)  # expand parameter path
		train_loader, test_loader = dataloader.loadSpaceshipIdentifierDataset(width, height)


	def getStatistics(self, path):
		pass


	def displayResults(self, game):
		pass