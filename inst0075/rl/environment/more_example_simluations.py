import numpy as np

from fomlads.rl.environment.simulation import Simulation

class ForagingSim(Simulation):

    def __init__(self, height, width):
        self.height = height
        self.width = width
        # now call the __init__ function on the superclass
        super().__init__(None, None, None, None) 

    def reset(self):
        self.game_over = False
        self.food_grid = np.ones((width, height), dtype=int)
        self.agent_loc = np.array((0,0))
        self.food_grid[self.agent_loc] = 0

    def next(self, action):
        """
        action 0: right, 1: down, 2: left, 3:up
        """
        # check that you don't move off grid
        #change location of agent
        if action == 0:
            self.agent_loc[1] = np.min(self.agent_loc[1]+1,self.width-1
        # do for other directions...
        if self.food_grid[self.agent_loc] == 1:
            reward = 1
            self.food_grid[self.agent_loc] = 0
        else:
            reward = 0

        if self.is_terminal():
            reward += 99
        next_state_rep = self.get_representation()
        return next_state_rep, reward

    def is_terminal(self):
        return np.all(self.food_grid == 0)

    def get_representation(self):
        rep = np.zeros(8, dtype=int)
        # above
        rep[0] = np.sum(self.food_grid[self.agent_loc[0]+1:,:])
        # above -right
        rep[1] = np.sum(self.food_grid[self.agent_loc[0]+1:,self.agent_loc[1]+1:])
        # right
        rep[2] = np.sum(self.food_grid[:,self.agent_loc[1]+1:])
        # below -right
        rep[3] = np.sum(self.food_grid[:max(self.agent_loc[0]-1,0),self.agent_loc[1]+1:])
        # and so on          
        return rep
