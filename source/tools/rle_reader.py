import re
import sys, os
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from game_of_life import Game

class RleReader:

	def __init__(self, filename=None, box=None):
		self.filename = filename
		self.box = box


	def getFileArray(self):
		if self.filename == None:
			return None

		with open(self.filename) as f:
			items = f.readlines()

		configurations = []
		if not items:
			return None

		currXYconfig = ()
		for item in items:
			if item[0] == "x":
				params = item.split(',')
				currXYconfig = (params[0].split("=")[1], params[1].split("=")[1])
				continue
			if self.box != None:
				configurations.append(self.placeConfigInBox(self.getConfig(item.rstrip(), int(currXYconfig[0]), int(currXYconfig[1])), self.box, self.box))
			else:
				configurations.append(self.getConfig(item.rstrip(), int(currXYconfig[0]), int(currXYconfig[1])))


		return configurations


	def getRleCodes(self):
		if self.filename == None:
			return None

		with open(self.filename) as f:
			items = f.readlines()

		if not items:
			return None

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


	def getConfig(self, code, x, y):
		grid = np.zeros((y, x))
		rle_list = code.split("$")

		for row, item in enumerate(rle_list):
			parseIndex = 0
			codeList = list(item)
			numStr = ""

			for i, item in enumerate(codeList):
				if item == 'o' or item == 'b':
					if numStr == "":
						grid[row, parseIndex] = int(item == 'o')
						parseIndex += 1

					else:
						grid[row, parseIndex:parseIndex+int(numStr)] = int(item =='o')
						parseIndex += int(numStr)
						numStr = ""

				elif item == "!":
					continue

				else:
					numStr += item

		return grid


if __name__ == "__main__":
	rleReader = RleReader("C:\\Workspace\\level-4-project\\source\\data\\30_30_all_spaceships.txt", box=30)
	items = rleReader.getFileArray()
	game = Game(30, 30)
	game.renderItemList(items)
	game.run()