import numpy as np

from inst0075.rl.environment.model.grid_world import GridWorld

def grid_world1(height=None, width=None):
    """
    Creates the simple grid-world seen in the lectures
    """
    if height is None:
        height = 4
    if width is None:
        width = 4
    # number of rows, columns in gridworld
    shape = (height, width)
    # actions N, E, S, W resolve deterministically
    action_effects = [1., 0, 0, 0]
    # there are no obstacles in this simple grid-world
    obstacles = []
    # absorbing/terminal locations
    absorbing_locs = [(0,0),(height-1, width-1)]
    # the default reward
    reward_default = -1
    # no locations with non-default rewards
    reward_special_locs = []
    # so no associated  values for the non-default rewards
    reward_special_vals = []
    # initial locations (agent can start anywhere except an absorbing location
    init_locs = [
        (i,j)
            for i in range(height)
                for j in range(width)
                    if not (i,j) in absorbing_locs ]
    # equal probability of any valid starting state
    init_weights = np.ones(len(init_locs))
    #
    gw = GridWorld.build(
        shape, obstacles, absorbing_locs, action_effects,  reward_default,
        reward_special_locs, reward_special_vals, init_locs, init_weights)
    return gw

def grid_world2(action_effects=None):
    # number of rows, columns in gridworld
    shape = (3,4)
    # probability of respectively moving along 0 degrees, 90 degrees
    # 180 degrees or 270 degrees from desired direction.
    if action_effects is None:
        action_effects = [0.8, 0.1, 0.0, 0.1]
    # obstacles block movement
    obstacles = [(1,1)]
    # absorbing/terminal locations
    absorbing_locs = [(0,3),(1,3)]
    # the default reward
    reward_default = -1
    # locations with non-default rewards
    reward_special_locs = [(0,3),(1,3)]
    # the values for the non-default rewards
    reward_special_vals = [10, -100]
    # initial locations
    init_locs = [(2,0)]
    # weights associated with non-zero probability for initial states
    init_weights = np.ones(len(init_locs))
    #
    gw = GridWorld.build(
        shape, obstacles, absorbing_locs, action_effects,  reward_default,
        reward_special_locs, reward_special_vals, init_locs, init_weights)
    return gw

def grid_world3(height=None, width=None):
    if height is None:
        height = 4
    if width is None:
        width = 12
    assert height >= 3, "Minimum height is 3"
    assert width >= 5, "Minimum width is 5"
    # number of rows, columns in gridworld
    shape = (height, width)
    # probability of respectively moving along 0 degrees, 90 degrees
    # 180 degrees or 270 degrees from desired direction.
    action_effects = [1, 0, 0, 0]
    # obstacles block movement
    obstacles = []
    # absorbing/terminal locations
    absorbing_locs = [ (height-1,j) for j in range(1,width)]
    # the default reward
    reward_default = -1
    # locations with non-default rewards
    reward_special_locs = absorbing_locs[:-1]
    # the values for the non-default rewards
    reward_special_vals = [-100] * (width-2)
    # initial locations
    init_locs = [(height-1,0)]
    # weights associated with non-zero probability for initial states
    init_weights = np.ones(len(init_locs))
    #
    gw = GridWorld.build(
        shape, obstacles, absorbing_locs, action_effects,  reward_default,
        reward_special_locs, reward_special_vals, init_locs, init_weights)
    return gw

def grid_world_the_cliff():
    return grid_world3(4,12)

def grid_world4():
    # number of rows, columns in gridworld
    shape = (5,5)
    # probability of respectively moving along 0 degrees, 90 degrees
    # 180 degrees or 270 degrees from desired direction.
    action_effects = [ 1, 0, 0, 0]
    # obstacles block movement
    obstacles = [(1, 1), (2, 1), (0, 3), (1, 3), (2, 3)]
    # absorbing/terminal locations
    absorbing_locs = [
        (2, 2), (2, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (2, 4)]
    # the default reward
    reward_default = 0
    # locations with non-default rewards
    reward_special_locs = absorbing_locs
    # the values for the non-default rewards
    reward_special_vals = [1, 10] + [-100]*5
    # initial locations 
    init_locs = [(3,0)]
    # weights associated with non-zero probability for initial states
    init_weights = np.ones(len(init_locs))
    #
    gw = GridWorld.build(
        shape, obstacles, absorbing_locs, action_effects,  reward_default,
        reward_special_locs, reward_special_vals, init_locs, init_weights)
    return gw


