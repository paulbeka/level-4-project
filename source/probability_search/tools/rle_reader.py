# TODO:
# - Make the getfromfilearray method take in a file as a paremeter

import re
import os
import numpy as np

class RleReader:

	def __init__(self, box=None):
		self.box = box


	def getFileArray(self, filename):
		items = self.getRleCodes(filename)
		configurations = []

		for item in items:
			if self.box != None:
				configurations.append(self.placeConfigInBox(self.getConfig(item), self.box, self.box))
			else:
				configurations.append(self.getConfig(item))

		return configurations


	def getRleCodes(self, filename):
		if not os.path.exists(filename):
			assert False, f"File {filename} does not exist."

		with open(filename, errors="ignore") as f:
			items = f.readlines()

		codes = []
		currCode = ""
		for i, item in enumerate(items):
			if i % 2:
				currCode += item.rstrip()
				codes.append(currCode)
				currCode = ""

			else:
				currCode += item

		return codes


	def placeConfigInBox(self, config, width, height):
		newBox = np.zeros((width, height))
		if width < config.shape[0] or height < config.shape[1]:
			return None

		begX, begY = (width//2) - (config.shape[0] // 2), (height//2) - (config.shape[1] // 2)

		aliveLoc = np.argwhere(config == 1)
		newBox[aliveLoc[:, 0]+begX, aliveLoc[:,1]+begY] = 1
		return newBox


	def getConfig(self, code):
		dims, code = code.split("\n")
		# get the grid dimentions from rle format: x=x,y=y
		x, y = dims.split(",")
		x = int(x.split("=")[1])
		y = int(y.split("=")[1])
		grid = np.zeros((x, y))

		# get the rest of the code
		rle_list = code.split("$")
		rowIndex = 0
		for row, item in enumerate(rle_list):
			parseIndex = 0
			codeList = list(item)
			numStr = ""

			for i, item in enumerate(codeList):
				if item == 'o' or item == 'b':
					if numStr == "":
						grid[parseIndex, rowIndex] = int(item == 'o')
						parseIndex += 1

					else:
						grid[parseIndex:parseIndex+int(numStr), rowIndex] = int(item =='o')
						parseIndex += int(numStr)
						numStr = ""

				elif item == "!":
					continue

				else:
					numStr += item

			if item[-1] != 'o' and item[-1] != '!':
				rowIndex += int(item[-1])
			else:
				rowIndex += 1

		return grid


if __name__ == "__main__":
	pass
	# rleReader = RleReader("C:\\Workspace\\level-4-project\\source\\data\\random_rle.txt", box=30)
	# items = rleReader.getFileArray()
	# game = Game(30, 30)
	# game.renderItemList(items)
	# game.run()