import numpy as np
import pygame


class Game:

	WHITE = (255, 255, 255)
	BLACK = (0, 0, 0)
	NEIGHBOUR_TEMPLATE = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
	DOUBLE_NEIGHBOUR_TEMPLATE = np.array(
			[[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1],
			[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], 
			[2, 2], [2, 1], [2, 0], [2, 1], [2, 2],
			[-1, 2], [0, 2], [1, 2], [-1, -2], [0, -2], [1, -2], [2, -2], [2, -1]]
		)

	def __init__(self, 
				width, 
				height, 
				rule1=[2, 3], 
				rule2=[3], 
				randomStart=False, 
				show_display=True):

		pygame.init()

		self.cell_size = 5
		self.width, self.height = width*self.cell_size, height*self.cell_size
		self.rule1, self.rule2 = rule1, rule2
		self.show_display = show_display

		# parameters
		self.fps = 60
		self.tick_update = 0
		self.generations_per_second = 2
		self.updateCellsAutomatically = True

		self.clock = pygame.time.Clock()
		self.running = True

		if self.show_display:
			self.display = pygame.display.set_mode((self.width, self.height))
			pygame.display.set_caption("Game of Life")

		self.x_size = self.width // self.cell_size
		self.y_size = self.height // self.cell_size

		# Generate random state
		if randomStart:
			self.cells = np.random.choice(a=[False, True], size=self.x_size * self.y_size)
		else:
			self.cells = np.zeros(self.x_size * self.y_size, dtype=np.bool_)

		self.cells = self.cells.reshape(self.x_size, self.y_size)

		# # preset cells - glider
		# self.cells[4, 4] = True
		# self.cells[5, 5] = True
		# self.cells[6, 5] = True
		# self.cells[6, 4] = True
		# self.cells[6, 3] = True

		# additional presets
		# self.cells[6, 6] = True
		# self.cells[7, 7] = True

		self.itemListToBeRendered = None
		self.itemListToBeRenderedIndex = 0


	def run(self):
		while self.running:
			self.update()
			if self.show_display:
				self.render()
		if self.show_display:
			pygame.display.quit()
		pygame.quit()


	def update(self):
		self.tick_update += 1

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False

			if event.type == pygame.KEYDOWN and self.renderNextItemList != None:
				if event.key == pygame.K_RIGHT:
					self.renderNextItemList(1)
				elif event.key == pygame.K_LEFT:
					self.renderNextItemList(-1)
				elif event.key == pygame.K_SPACE:
					self.updateCellsAutomatically = not self.updateCellsAutomatically


		if (self.fps // self.generations_per_second) < self.tick_update:
			self.tick_update = 0
			# update board state
			if self.updateCellsAutomatically:
				self.cells = self.getNextState()


	def render(self):
		self.display.fill(Game.WHITE)

		self.draw_grid()
		self.draw_cells()

		pygame.display.flip()
		self.clock.tick(self.fps)


	def draw_grid(self):
		for x in range(self.x_size):
			pygame.draw.line(self.display, Game.BLACK, (x*self.cell_size, 0), (x*self.cell_size, self.height))
		for y in range(self.y_size):
			pygame.draw.line(self.display, Game.BLACK, (0, y*self.cell_size), (self.width, y*self.cell_size))


	def draw_cells(self):
		indexes = np.argwhere(self.cells == 1)
		for index in indexes:
			pygame.draw.rect(self.display, Game.BLACK, ((index[0]*self.cell_size, index[1]*self.cell_size), (self.cell_size, self.cell_size)), False)


	def getNextState(self):
		active_cells = np.argwhere(self.cells == 1)
		update = active_cells

		newState = np.array(self.cells, copy=True)

		for cell in active_cells:
			neighbors = Game.NEIGHBOUR_TEMPLATE + cell
			update = np.append(update, neighbors, axis=0)

		for cell in update:
			# cell out of bounds

			if cell[0] < 0 or cell[0] > self.x_size-1 or cell[1] < 0 or cell[1] > self.y_size-1:
				continue

			n_live_neighbours = self.checkNumberOfNeighbours(cell)

			# Rules of the game
			if self.cells[cell[0], cell[1]]:
				if not n_live_neighbours in self.rule1:
					newState[cell[0], cell[1]] = False

			else:
				if n_live_neighbours in self.rule2:
					newState[cell[0], cell[1]] = True

		return newState		


	# cell checking functions 
	def checkNumberOfNeighbours(self, cell):
		neighbors = self.getValidNeighbours(cell)
		return np.count_nonzero(self.cells[neighbors[:,0],neighbors[:,1]] == 1)


	def getValidNeighbours(self, cell):
		neighbors = Game.NEIGHBOUR_TEMPLATE + cell
			
		# Check that neighbours are within the board
		valid_neighbours = neighbors[neighbors[:,0] > -1]
		valid_neighbours = valid_neighbours[valid_neighbours[:,0] < self.x_size]
		valid_neighbours = valid_neighbours[valid_neighbours[:,1] > -1]
		valid_neighbours = valid_neighbours[valid_neighbours[:,1] < self.y_size]

		return valid_neighbours


	# evolve a specific configuration
	def evolve(self, board):
		# change this temp x solution
		temp = self.x_size, self.y_size
		self.x_size, self.y_size = board.shape
		self.cells = board
		x = self.getNextState()
		self.x_size, self.y_size = temp
		return x


	# find the pattern identity of an object
	def patternIdentity(self, pattern):
		modifiedPattern = pattern.copy()
		if pattern.shape[1] != 2:
			modifiedPattern = np.argwhere(pattern == 1)
		reference = (min(modifiedPattern[:,0]), min(modifiedPattern[:,1]))
		modifiedPattern[:, 0] -= reference[0]
		modifiedPattern[:, 1] -= reference[1]
		return modifiedPattern


	### RENDERING MISC ITEMS ###
	def renderItemList(self, itemList):
		if not itemList:
			return False
		self.itemListToBeRendered = itemList
		self.updateCellsAutomatically = False
		self.renderNextItemList(0)


	def renderNextItemList(self, direction):
		self.itemListToBeRenderedIndex += direction

		# loop around the list
		if self.itemListToBeRenderedIndex > len(self.itemListToBeRendered) - 1:
			self.itemListToBeRenderedIndex = 0
		elif self.itemListToBeRenderedIndex < 0:
			self.itemListToBeRenderedIndex = len(self.itemListToBeRendered) - 1

		# check to see if list has an information attribute to be printed
		if len(self.itemListToBeRendered[self.itemListToBeRenderedIndex]) > 1:
			if not isinstance(self.itemListToBeRendered[self.itemListToBeRenderedIndex][0], np.ndarray):
				self.cells = np.array(self.itemListToBeRendered[self.itemListToBeRenderedIndex][0])
			else:
				self.cells = self.itemListToBeRendered[self.itemListToBeRenderedIndex][0]
				
			print(self.itemListToBeRendered[self.itemListToBeRenderedIndex][1])
		else:
			self.cells = self.itemListToBeRendered[self.itemListToBeRenderedIndex][0]


	def kill(self):
		self.running = False


def main():
	game = Game(20, 20, randomStart=False)
	game.run()
	pygame.quit()


if __name__ == "__main__":
	main()