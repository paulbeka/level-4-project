import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# from networks.probability_finder import ProbabilityFinder
from networks.convolution_probability_network import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions


### HYPERPARAMETERS ###
num_epochs = 3
batch_size = 1
learning_rate = 0.0001
n_errors_per_spaceship = 10

model =  ProbabilityFinder(batch_size).double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


### DATA LOADING ###
train_loader, test_loader = getPairSolutions(0.8, n_errors_per_spaceship, batch_size, "advanced_deconstruct")  # n_pairs : fake data for every ship
print("Data loaded.")


### NEURAL NET ###
total_steps = len(train_loader)
for epoch in range(num_epochs):
	for i, (configs, labels) in enumerate(train_loader):

		outputs = model(configs)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 10000 == 0:
			print(f"Loss: {loss.item():.9f}")
			print(f"{i}/{len(train_loader)} datasets done")

	print(f"Epoch: {epoch+1}/{num_epochs}")


### TESTING ###
with torch.no_grad():
	correct, samples = 0, 0
	test_loss = 0
	for configs, labels in test_loader:
		outputs = model(configs)

		_, predictions = torch.max(outputs, 1)
		test_loss += criterion(outputs, labels)

		samples += labels.shape[0]
		correct += (predictions == labels).sum().item()

	accuracy = 100 * (correct / samples)
	print(f"Accuracy: {accuracy:.2f}%")
	print(f"Total loss: {test_loss / len(test_loader)}")

	# SAVE THE MODEL TO A FILE
	save_model = True  # set to true if you wish model to be saved
	save_name = ""
	if save_model:
		save_name = input("Input the name of your NN: ")

	SAVE_PATH = os.path.join(os.getcwd(), "models", save_name)

	if save_model:
		torch.save(model.state_dict(), SAVE_PATH)
