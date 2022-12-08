import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

from game_of_life import Game
from tools.rle_reader import RleReader
from dataloaders.spaceshipid_dataloader import SpaceshipIdentifierDataLoader
from dataloaders.spaceship_comparisons_dataloader import SpaceshipCompareDataloader
from networks.lifenet_cnn import LifeNetCNN


### CONSTANTS ###
DATA_PATH = os.path.join(os.getcwd(), "data")


### HYPERPARAMETERS ###
num_epochs = 50
batch_size = 1
learning_rate = 0.0001

save_model = True  # set to true if you wish model to be saved
save_name = ""
if save_model:
	save_name = input("Input the name of your NN: ")
SAVE_PATH = os.path.join(DATA_PATH, "models", save_name)

width, height = 100, 100

### LOAD DATA ###

dataloader_params = {'dataloader': [
										1000, 
										0.5, 
										True, 
										os.path.join(DATA_PATH, "spaceship_identification"), 
										1, 
										False
									],
					'width' : width,
					'height': height,
					'num_epochs' : num_epochs,
					'batch_size' : batch_size
					}

dataloader_params = {'dataloader': [
										1000,  
										os.path.join("C:\\Workspace\\level-4-project\\source\\data", "spaceship_identification"), 
										1
									],
					'width' : width,
					'height': height,
					'num_epochs' : num_epochs,
					'batch_size' : batch_size
					}

#dataloader = SpaceshipIdentifierDataLoader(*dataloader_params['dataloader'])
dataloader = SpaceshipCompareDataloader(*dataloader_params['dataloader'])

train_loader, test_loader = dataloader.loadData(width, height)

### NEURAL NET ###

num_classes = 2

model = LifeNetCNN(num_classes, batch_size).double()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (configs, labels) in enumerate(train_loader):
		outputs = model(configs)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 100 == 0:
			print(f"Loss: {loss.item():.9f}")

	print(f"Epoch: {epoch+1}/{num_epochs}")


### TESTING ###

with torch.no_grad():
	correct, samples = 0, 0
	for configs, labels in test_loader:
		outputs = model(configs)

		_, predictions = torch.max(outputs, 1)

		samples += labels.shape[0]
		correct += (predictions == labels).sum().item()

	accuracy = 100 * (correct / samples)
	print(f"Accuracy: {accuracy:.2f}%")

	# save items into the data/models folder
	if save_model:
		torch.save(model.state_dict(), SAVE_PATH)

		with open(os.path.join(DATA_PATH, "models", save_name + "_parameters.txt"), "w") as f:
			json.dump(dataloader_params, f)


# run
if __name__ == '__main__':
	pass