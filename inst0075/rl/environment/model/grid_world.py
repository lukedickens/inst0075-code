import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from inst0075.rl.environment.model.mdp import MDP

class GridWorld(MDP):
    """
    Creates objects that represent grid-worlds
    """
    # the cardinal points of the compass encoded as integers
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    DIRECTIONS = [ NORTH, EAST, SOUTH, WEST ]

    @classmethod
    def build(
            cls, shape, obstacles, absorbing_locs, action_effects,
            reward_default, reward_special_locs, reward_special_vals,
            init_locs, init_weights ):
        """
        Builds a grid world MDP model given a description in terms of a grid
        shape a list of obstacles (invalid grid locations) absorbing locations
        (terminal states), action effects, reward and initialisation
        conditions.

        parameters
        ----------
        shape - (height, width) of the underlying grid
        obstacles - a list of tuples where the nth tuple is the (i,j)th grid
          position of the nth invalid position    
        absorbing_locs - a list of tuples: each tuple is the (i,j) grid
            location of an absorbing grid-position
        init_locations - grid positions of locations with non-zero probs
        init_weights - weights/probabilities of initial locations

        returns
        -------
        A grid world object 
          which is an mdp model of the grid-world with the following additional
          attributes

              shape - the shape (height, width) of the grid-world
              locs - the mapping from states to locations (i,j)
              neighbours - an Nx4 array, where the (n,k)th element is the  state
                  index of the neigbouring state in the kth direction

        """
        locs, neighbours = cls.get_topology(shape,obstacles)
        #
        ## how many states and actions are there
        num_states = len(locs)
        # we hard code 4 actions: N, E, S, W
        action_names = ['N', 'E', 'S', 'W']
        num_actions = len(action_names)
        #
        ## absorbing_locs is list of locations one per absorbing state
        ## absorbing is a boolean array with 1s for all absorbing state indices
        absorbing_states = [ locs.index(ab_loc) for ab_loc in absorbing_locs ]
        absorbing = np.zeros(num_states, dtype=bool)
        absorbing[absorbing_states] = True
        #
        ## initial is array of probabilities representing the intial state
        # distribution
        init_locs = init_locs
        init_weights = init_weights
        initial_states = [ locs.index(i_loc) for i_loc in init_locs ]
        initial = np.zeros(num_states)
        initial[initial_states] = init_weights
        initial /= np.sum(initial)
        
        #
        ## Notation:
        ##      s = prior state
        ##      s_ = posterior state
        #
        ## build the transition matrix, each slice T[a,:,:] is the state
        # transition matrix under the action a
        T = np.zeros((num_actions, num_states, num_states))
        for a in range(num_actions):
            for effect in range(num_actions):
                # the outcome direction is the action modified by the
                # non-deterministic effect. An effect of 0 takes the agent in 
                # direction chosen, 1 takes the agent 90 degrees clockwise of
                #  that and so on.
                outcome = (a+effect) % num_actions
                prob = action_effects[effect]
                for s in range(num_states):
                    s_ = neighbours[s, outcome]
                    T[a, s, s_] = T[a, s, s_] + prob
        #
        #
        ## build the reward matrix
        R = reward_default*np.ones((num_actions, num_states, num_states))
        if len(reward_special_locs) != 0:
            # if there are special locations
            # get the state index of each special location
            special_states = [locs.index(loc) for loc in reward_special_locs ]
            # update the reward matrix so that all transitions ending in the
            # appropriate state have the right reward (reward only depends on
            # posterior state).
            # NOTE: There is redundancy in this matrix (some transitions never
            # occur) we do not care what reward value these transitions have
            for s, r in zip(special_states, reward_special_vals):
                R[:,:,s] = r
        #
        # create the grid world object itself and return it
        return cls(num_states, action_names, initial, absorbing,
            T, R, shape, locs, neighbours, obstacles)

    def __init__(self, num_states, action_names, initial, absorbing,
            T, R, shape, locs, neighbours, obstacles):
        """
        The constructor for a grid-world. It is recommended to use the 
        class method build to build grid worlds.
        """
        # the core of the grid world is the mdp model
        super().__init__(num_states, action_names, initial, absorbing,
            T_matrix=T, R_matrix=R)
        # additional components for grid world models
        # the shape (height, width) of the grid-world
        self.shape = shape
        # the mapping from states to locations (i,j)
        self.locs = locs
        # mapping from each state to the 4 neighbouring states (aka topology)
        self.neighbours = neighbours
        # the obstacles
        self.obstacles = obstacles

    def get_feature_mapping_tiling(
            self,tile_size=3, starting_positions=[[0,0],[2,1],[1,2]]):
        locs = np.array(self.locs)
        tile_cols = math.ceil(self.shape[0]/tile_size)
        tile_rows = math.ceil(self.shape[1]/tile_size)
        # each state sits within a tile index one per tiling
        # first we calculate these tiling specific indices
        state_indices = np.zeros(
            (self.num_states, len(starting_positions)), dtype=int)
        for tiling, (xshift, yshift) in enumerate(starting_positions):
            xs = locs[:,0]
            ys = locs[:,1]
            indices = tile_cols * ((ys-yshift)/tile_size).astype(int) \
                + ((xs-xshift)/tile_size).astype(int)
            state_indices[:,tiling] = indices
        # the maximum number of indices per tiling help us to construct the
        # full features
        subfeature_lengths = np.max(state_indices, axis=0) + 1
        # each state's feature is a stack of one-hot-vectors one per tiling 
        # whose non-zero element is at the index of the state in that tiling.
        features = np.zeros(
            (self.num_states, 1+np.sum(subfeature_lengths)),dtype=int)
        # the zeroth feature is the constant term
        features[:,0] = 1
        min_index = 1
        for tiling, sub_len in enumerate(subfeature_lengths):
            # for every state get the local index and add the min_index
            these_indices = min_index + state_indices[:,tiling]
            features[np.arange(self.num_states), these_indices] = 1
            min_index += sub_len
        def feature_mapping(state):
            return features[state,:]
        return feature_mapping

    def get_feature_mapping_xy(self, const_feature=True):
        features = np.array(self.locs)
        if const_feature:
            ones = np.ones((features.shape[0],1))
            features = np.hstack(
                (ones, features))
        def feature_mapping(state):
            return features[state,:]
        return feature_mapping

    def get_feature_mapping_neighbours_binary(self, const_feature=True):
        locs = np.array(self.locs)
        # access grid is a boolean matrix of the grid world with a margin
        # width 1 all the way around corresponding to the inaccessible 
        # region round the edge (all False). Within the grid proper
        # accessible cells are True and inaccessible (obstacles) are False
        access_grid = np.ones((self.shape[0]+2,self.shape[1]+2), dtype=int)
        access_grid[1:-1,1:-1] = np.zeros(self.shape, dtype=int)
        for i, j in self.obstacles:
            access_grid[i+1, j+1] = True
        # need mask to remove the middle square of the flattened feature
        # (which is accesible by definition)
        mask = np.ones(9,dtype=bool)
        mask[4] = False
        features = np.empty((self.num_states, 8), dtype=int)
        for s, (i, j) in enumerate(locs): 
            # 3x3 grid of accessibility for cell i,j
            neighbours = access_grid[i:i+3, j:j+3]
            features[s,:] = neighbours.flatten()[mask]
        if const_feature:
            featurevec = np.hstack(
                (np.ones((self.num_states,1),dtype=int),features))
        def feature_mapping(state):
            return features[state,:]
        return feature_mapping

    def get_feature_mapping_neighbours_enumerated(self):
        binary_mapping = self.get_feature_mapping_neighbours_binary(
            const_feature=False)
        def feature_mapping(state):
            neighbours = binary_mapping(state)
            print("neighbours = %r" % (neighbours,) )
            index = np.sum(2**np.arange(8) * neighbours)
            featurevec = np.zeros(256, dtype=int)
            featurevec[index] = 1
            return featurevec
        return feature_mapping

    @classmethod
    def neighbour_index(cls, neighbours):
        index = 0
        power = 0
        for i, row in enumerate(neighbours):
            for j, val in enumerate(row):
                if not (i == j == 1):
                    if neighbours[i,j]:
                        index += 2**power
                    power += 1
        return index

    @classmethod
    def get_topology(cls, shape, obstacles):
        """ 
        shape - (width, height) of grid 
        obstables - a list of (i,j) pairs corresponding to grid positions that
            are not valid states
        """
        height, width = shape
        locs = []
        all_neighbour_locs = []
        index = 1
        for i in range(height):
            for j in range(width):
                loc = (i,j)
                # for every potential grid-position we only add it to the list
                # of locs if it is a valid location
                if cls.valid_location(loc, shape, obstacles):
                    locs.append(loc)
                    all_neighbour_locs.append(
                        cls.get_neighbour_list(loc, shape, obstacles))
        # translate neighbour lists from locations to states
        num_states = len(locs)
        num_actions = len(cls.DIRECTIONS)
        neighbours = np.empty((num_states,4), dtype=int)
        for s, these_neighbours in enumerate(all_neighbour_locs):
            for dirn in range(num_actions):
                neighbour_loc = these_neighbours[dirn]
                # find index of neighbour location
                # to turn location into a state number
                neighbour_state = locs.index(neighbour_loc)
                # insert into neighbour matrix
                neighbours[s,dirn] = neighbour_state
        #
        return locs, neighbours

    @classmethod
    def get_neighbour_list(cls, loc, shape, obstacles):
        """
        Given a location in our grid-world, this determines the ordered list of
        neighbouring locations (one per direction). 

        parameters
        ----------
        loc - (i,j) grid location
        shape - shape of the grid-world
        obstacles - list of grid-position that are not valid due to obstacles

        returns
        -------
        neighbour_list - neighbouring grid-positions, or loc when neighbour is
            invalid
        """
        neighbour_list = []
        for dirn in cls.DIRECTIONS:
            this_neighbour = cls.get_neighbour(loc, dirn, shape, obstacles)
            neighbour_list.append(this_neighbour)
        return neighbour_list

    @classmethod
    def get_neighbour(cls, loc, dirn, shape, obstacles):
      """
      Given a location in our grid-world and a direction, this determines the
      neighbouring location if there is one, if not it returns the input
      location.

      Used to determine where agent ends up if moving in a given direction
      from a given location.

      parameters
      ----------
      loc - (i,j) grid location
      dirn - NORTH, EAST, SOUTH or WEST (cardinal DIRECTIONS)
      shape - shape of the grid-world
      obstacles - list of grid-position that are not valid due to obstacles

      returns
      -------
      neighbour - neighbouring grid-position in the direction specified or loc
          if the direction is invalid
      """
      i, j = loc
      if dirn == cls.NORTH:
          target = (i-1,j)
      elif dirn == cls.EAST:
          target = (i,j+1)
      elif dirn == cls.SOUTH:
          target = (i+1,j)
      elif dirn == cls.WEST:
          target = (i,j-1)
      else:
        raise ValueError("Unrecognised direction %d" % dirn)
      # target refers to the grid position that the agent would move to if it
      # were a valid location, if it is not valid then return the present
      # location
      if cls.valid_location(target, shape, obstacles):
          neighbour = target
      else:
          neighbour = loc
      return neighbour

    @classmethod
    def valid_location(cls, loc, shape, obstacles):
      """
      Evaluates whether a grid location is valid for this grid world

      parameters
      ----------
      loc - pair of x, y grid coordinates
      shape - (width, height) of grid world
      obstacles - a list of grid coordinates that are forbidden locations
      """
      x, y = loc
      max_x, max_y = shape
      if x<0 or y<0 or x>=max_x or y >= max_y :
        return False
      elif loc in obstacles:
        return False
      return True

    def visualise(
            self, labels=None, suppress_absorbing=False, fontsize=16):
        """
        Plots a visualisation of a grid world. Cells are coloured according
        to type good absorbing (green), bad absorbing (red), valid (blue), and
        invalid grid locations are greyed out. Labels are printed in text
        within each cell (only in non-absorbing cells if absorbing suppressed).

        parameters
        ----------
        labels (optional) - a list or vector of string labels or numeric 
            values. If no labels are provided then cells will be labelled with
            state names.
        suppress_absorbing (optional) - if set to true then do not print label
            in locations corresponding to absorbing states
        fontsize (optional) - size of label font

        returns
        -------
        fig - matplotlib.Figure object
        ax - matplotlib.Axes object
        """
        # define colors to visually represent grid locations by cell type
        c_background = [0.2, 0.2, 0.2]
        v_background = 0
        c_loc = [0.8, 0.9, 1]
        v_loc = 1
        c_bad = [1,0.7,0.7]
        v_bad = 2
        c_good = [0.7,1,0.7]
        v_good = 3
        cmap = matplotlib.colors.ListedColormap(
            [c_background, c_loc, c_bad,  c_good])
        bounds = np.arange(v_good+2)-0.5
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        #
        # cells is a grid of integers corresponding to cell type: 1 (normal), 
        # 0 (invalid), 2 (terminal and bad) and 3 (terminal and good)
        cells = v_background*np.ones(self.shape, dtype=int)
        locs = np.array(self.locs)
        for i, j in locs:
            cells[i,j] = v_loc
        absorbing_locs = locs[self.absorbing,:]
        state_rewards = self.R[0,0,:]
        mean_reward = np.mean(state_rewards)
        final_rewards = state_rewards[self.absorbing]
        for (i,j), r in zip(absorbing_locs, final_rewards):
            if r >= mean_reward:
                cells[i,j] = v_good
            else:
                cells[i,j] = v_bad
        #
        # now we can plot the grid and colour the cells accordingly
        fig, ax = plt.subplots(figsize=(self.shape[1], self.shape[0]))
        ax.imshow(cells, cmap=cmap, norm=norm)
        #
        # Labels:
        # write label in each cell (convert floats to nicely formatted strings)
        # if labels is not defined then use state names
        if labels is None:
            labels = self.state_names
        # if labels are real numbers format them nicely
        elif isinstance(labels[0], (float, np.float32, np.float64)):
            labels = map(lambda l: '%.2g' % l, labels)
            fontsize -= 2
        # for each valid location print the label
        for (col,row), label, is_absorbing in zip(
                locs, labels, self.absorbing):
            # don't label absorbing states if they are suppressed
            if suppress_absorbing and is_absorbing:
                continue
            ax.text(row,col, label, horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
        # include a nice border for cells
        ax.set_xticks(np.arange(self.shape[1]+1)-0.5)
        ax.set_yticks(np.arange(self.shape[0]+1)-0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(
            which='major', axis='both', linestyle='-', color=c_background,
            linewidth=2)
        #
        return fig, ax



