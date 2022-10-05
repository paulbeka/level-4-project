import numpy as np
import pygame


class Game:

	WHITE = (255, 255, 255)
	BLACK = (0, 0, 0)

	def __init__(self, width, height, rule1=[2, 3], rule2=[3], randomStart=False):

		pygame.init()

		self.width, self.height = width, height
		self.rule1, self.rule2 = rule1, rule2

		# parameters
		self.show_display = True
		self.fps = 60
		self.tick_update = 0
		self.generations_per_second = 60

		self.clock = pygame.time.Clock()
		self.running = True
		if self.show_display:
			self.display = pygame.display.set_mode((self.width, self.height))


		self.cell_size = 20
		self.x_size = self.width // self.cell_size
		self.y_size = self.height // self.cell_size

		# Generate random state
		if randomStart:
			self.cells = np.random.choice(a=[False, True], size=self.x_size * self.y_size)
		else:
			self.cells = np.zeros(self.x_size * self.y_size, dtype=np.bool_)
		
		self.cells = self.cells.reshape(self.x_size, self.y_size)
	
		self.neighbour_template = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
		self.neighbour_template = np.array(self.neighbour_template)

		# preset cells - glider
		self.cells[4, 4] = True
		self.cells[5, 5] = True
		self.cells[6, 5] = True
		self.cells[6, 4] = True
		self.cells[6, 3] = True

		# additional presets
		self.cells[6, 6] = True
		self.cells[7, 7] = True

	def run(self):
		while self.running:
			self.update()
			if self.show_display:
				self.render()


	def update(self):
		self.tick_update += 1

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False

		if self.fps // self.generations_per_second < self.tick_update:
			self.update_cells()
			self.tick_update = 0


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


	def update_cells(self):
		active_cells = np.argwhere(self.cells == 1)
		update = active_cells

		newState = np.array(self.cells, copy=True)

		for cell in active_cells:
			neighbors = self.neighbour_template + cell
			update = np.append(update, neighbors, axis=0)

		for cell in update:
			# cell out of bounds

			if cell[0] < 0 or cell[0] > self.x_size-1 or cell[1] < 0 or cell[1] > self.y_size-1:
				continue

			neighbors = self.neighbour_template + cell
			
			# Check that neighbours are within the board
			valid_neighbours = neighbors[neighbors[:,0] > -1]
			valid_neighbours = valid_neighbours[valid_neighbours[:,0] < self.x_size]
			valid_neighbours = valid_neighbours[valid_neighbours[:,1] > -1]
			valid_neighbours = valid_neighbours[valid_neighbours[:,1] < self.y_size]

			n_live_neighbours = np.count_nonzero(self.cells[valid_neighbours[:,0],valid_neighbours[:,1]] == 1)
			#print(n_live_neighbours)

			# Rules of the game
			if self.cells[cell[0], cell[1]]:
				if not n_live_neighbours in self.rule1:
					newState[cell[0], cell[1]] = False

			else:
				if n_live_neighbours in self.rule2:
					newState[cell[0], cell[1]] = True

		self.cells = newState


	def getState(self):
		return self.cells


def main():
	game = Game(800, 800, randomStart=False)
	game.run()
	pygame.quit()


if __name__ == "__main__":
	main()