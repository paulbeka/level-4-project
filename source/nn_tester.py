import torch
import os
import json

from networks.lifenet_cnn import LifeNetCNN
from dataloaders.spaceshipid_dataloader import SpaceshipIdentifierDataLoader
from dataloaders.spaceship_comparisons_dataloader import SpaceshipCompareDataloader
from game_of_life import Game


# TODO: load the batch_size and the num_classes from parameter list
class NeuralNetworkTester:

	def __init__(self, Network, DataLoader, path):

		# basic structure: all the parameters are kept in the same file as the model
		with open(path + "_parameters.txt") as f:
			params = json.load(f)

		self.width, self.height = params['width'], params['height']

		self.model = Network(2, 1).double()
		self.model.load_state_dict(torch.load(path))
		self.model.eval()
		
		dataloader = DataLoader(*params['dataloader'])  # expand parameter path
		self.train_loader, self.test_loader = dataloader.loadData(self.width, self.height)


	def getStatistics(self):
		resultList = []
		correct, samples = 0, 0

		for configs, labels in self.test_loader:
			outputs = self.model(configs)

			_, predictions = torch.max(outputs, 1)

			samples += labels.shape[0]
			correct += (predictions == labels).sum().item()

			resultList += [item for item in zip(configs, predictions)]


		accuracy = 100 * (correct / samples)
		stats = {
			'resultList' : resultList,
			'accuracy' : accuracy
		}

		return stats


	def gameDisplayResults(self):

		stats = self.getStatistics()

		results = stats['resultList']

		game = Game(self.width, self.height)
		game.renderItemList(results)
		game.run()


if __name__ == '__main__':
	# tester = NeuralNetworkTester(LifeNetCNN, 
	# 	SpaceshipIdentifierDataLoader,
	# 	"C:\\Workspace\\level-4-project\\source\\data\\models\\test")

	tester = NeuralNetworkTester(LifeNetCNN, SpaceshipCompareDataloader, "C:\\Workspace\\level-4-project\\source\\data\\models\\test")

	tester.gameDisplayResults()