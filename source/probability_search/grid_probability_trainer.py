import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

# from networks.probability_finder import ProbabilityFinder
from networks.convolution_probability_network import ProbabilityFinder
from dataloaders.probability_grid_dataloader import getPairSolutions


if not torch.cuda.is_available():
	print("GPU IS NOT AVAILABLE AND HAS BEEN IMPROPERLY CONFIGURED.")
	print("INSTALL THE NVIDIA DRIVER AND RETRY.")
	print("EXITING.")
	quit()


### HYPERPARAMETERS ###
num_epochs = 10
batch_size = 1
learning_rate = 0.00005
n_errors_per_spaceship = 15

model =  ProbabilityFinder(batch_size).double()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
# model.to(device)
model.cuda(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


### DATA LOADING ###
train_loader, test_loader = getPairSolutions(0.8, n_errors_per_spaceship, batch_size, "advanced_deconstruct")  # n_pairs : fake data for every ship
print("Data loaded.")


### NEURAL NET ###
total_steps = len(train_loader)
for epoch in range(num_epochs):
	print(f"Epoch: {epoch+1}/{num_epochs}")
	for i, (configs, labels) in tqdm(enumerate(train_loader), desc="Training: ", total=len(train_loader)):
		# load data into GPU
		configs, labels = configs.to(device), labels.to(device)
		
		outputs = model(configs)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print(f"Saving epoch {epoch+1}...")

	torch.save(model.state_dict(), f"outputFile_{epoch+1}")


### TESTING ###
# with torch.no_grad():
	# correct, samples = 0, 0
	# test_loss = 0
	# for configs, labels in test_loader:
	# 	outputs = model(configs)

	# 	_, predictions = torch.max(outputs, 1)
	# 	test_loss += criterion(outputs, labels)

	# 	samples += labels.shape[0]
	# 	correct += (predictions == labels).sum().item()

	# accuracy = 100 * (correct / samples)
	# # print(f"Accuracy: {accuracy:.2f}%")
	# # print(f"Total loss: {test_loss / len(test_loader)}")

	# # SAVE THE MODEL TO A FILE
	# save_model = True  # set to true if you wish model to be saved
	# save_name = ""
	# if save_model:
	# 	# save_name = input("Input the name of your NN: ")
	# 	save_name = "OUTPUT_SEND_THIS_BY_EMAIL"

	# SAVE_PATH = os.path.join(os.getcwd(), "models", save_name)

	# # if save_model:
	# # 	torch.save(model.state_dict(), SAVE_PATH)

print("EXECUTION OVER.")
