import numpy as np

from fomlads.rl.environment.model.mdp import MDP


def stair_climbing_mdp():
    """
      STAIRCLIMBINGMDP Creates a model of the simple stair climbing MDP from
      the lectures
    """
    # States are:  {P <-- s1 <=> s2 <=> s3 <=> s4 <=> s5 --> G]
    num_states = 7
    state_names =  ['P', 's1', 's2', 's3', 's4', 's5', 'G']

    # Actions are: [L,R]
    num_actions = 2
    action_names =  ['L', 'R']

    # Array indicating absorbing states
    #            P,   1,   2,   3,   4,   5,    G   <-- STATES 
    absorbing = np.array(
              [1,   0,   0,   0,   0,   0,   1], dtype=bool)
    # must be boolean

    # Array indicating starting state distribution
    #            P,   1,   2,   3,   4,   5,    G   <-- STATES 
    initial = [  0,   0,   0,   1,   0,   0,   0]

    # the transition function
    def stair_climbing_transition_function(s, a, s_):
        # we assume that transition's from absorbing states are to
        # the same state irrespective of action
        if s == 0 or s == 6:
            return int(s==s_)
        # action L always steps once to the left
        elif a == 0:
            return int((s-1)==s_)
        # action R always steps once to the right
        else: 
            return int((s+1)==s_)

    # the reward function
    def stair_climbing_reward_function(s, a, s_):
        # any subsequent action from the absorbing states get 0 reward
        if s == 0 or s == 6:
            return 0
        # moving to P
        if (s == 1) and (a == 0) and (s_ == 0):
            return -10.0
        # moving to G
        elif (s == 5) and (a == 1) and (s_ == 6):
            return 10.0
        # stepping left (action is L)
        elif (a == 0):
            return 1.0
        # stepping right (action is R)
        else:
            return -1.0

    mdp = MDP(
        state_names, action_names, initial, absorbing,
        t_function=stair_climbing_transition_function,
        r_function=stair_climbing_reward_function)

    return mdp


def stop_go_mdp():
    """
      Creates a model of the stop-go MDP from the tutorial
    """
    # States are:  {P <-- s1 <=> s2 <=> s3 <=> s4 <=> s5 --> G]
    state_names =  ['1', '2' ]

    # Actions are: [L,R]
    action_names =  ['stop', 'go']

    # Array indicating absorbing states
    absorbing = [False, False]

    # Array indicating starting state distribution
    initial = [1, 1]

    # the transition function
    def stop_go_t_func(s, a, s_):
        # a == 0 is stop, state remains the same w.p. 1
        if a == 0:
            return int(s==s_)
        # a == 1 is go, state changes w.p. 1
        else:
            return int(s!=s_)

    # the reward function
    def stop_go_r_func(s, a, s_):
        if a == 0 and s == 1:
            return 1
        if a == 1 and s == 2:
            return 2
        else:
            return 0

    mdp = MDP(
        state_names, action_names, initial, absorbing,
        t_function=stop_go_t_func, r_function=stop_go_r_func)

    return mdp
