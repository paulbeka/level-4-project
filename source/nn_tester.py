import torch
import os


class NeuralNetworkTester:

	def __init__(self, Network, path):
		self.model = Network()
		self.load_state_dict(torch.load(path))
		self.model.eval()

		# basic structure: all the parameters are kept in the same file as the model
		with open(path + "_parameters.txt") as f:
			params = [item.rstrip() for item in f.readlines()]

		dataloader = DataLoader(*params)  # expand parameter path
		self.train_loader, self.test_loader = dataloader.loadData(self.width, self.height)


	def getStatistics(self, path):
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


	def gameDisplayResults(self, game):

		stats = self.getStatistics()

		results = stats['resultList']

		game.renderItemList(results)
		game.run()

