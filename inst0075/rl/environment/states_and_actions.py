import numpy as np
import copy
import matplotlib.pyplot as plt

def choose_from_policy(policy, state):
    num_actions = policy.shape[1]
    return np.random.choice(num_actions, p=policy[state,:])


def get_epsilon_greedy_policy(epsilon, Q, absorbing=None):
    """
    Returns an epsilon-greedy policy from a Q-function estimate

    parameters
    ----------
    epsilon - should be 0<epsilon<0.5. This is the variable that controls the
        degree of randomness in the epsilon-greedy policy.
    Q - (num_states x num_actions) matrix of Q-function values
    absorbing (optional) - A vector of booleans, indicating which states are
        absorbing (and hence do not need action decisions). if specified then
        the rows of the output policy will not specify a probability vector
        
    returns
    -------
    policy - (num_states x num_actions) matrix of state dependent action
        probabilities.
    """
    num_actions = Q.shape[1]
    greedy_policy = get_greedy_policy(Q, absorbing=absorbing)
    policy = (1-epsilon)*greedy_policy + epsilon*np.ones(Q.shape)/num_actions
    return policy

def get_greedy_policy(Q, absorbing=None):
    """
    Returns the greedy policy from a Q-function estimate

    parameters
    ----------
    Q - (num_states x num_actions) matrix of Q-function values
    absorbing (optional) - A vector of booleans, indicating which states are
        absorbing (and hence do not need action decisions). if specified then
        the rows of the output policy will not specify a probability vector
        
    returns
    -------
    policy - (num_states x num_actions) matrix of state dependent action
        probabilities. However this will contain just one 1 per row with
        all other values zero. If a vector specifying absorbing states is 
        pass in then the corresponding rows will not be a valid probability
        vector
    """
    num_states, num_actions = Q.shape
    dominant_actions = np.argmax(Q, axis=1)
    policy = np.zeros((num_states, num_actions))
    policy[np.arange(num_states), dominant_actions] = 1.
    if not absorbing is None:
        # np.nan means Not-a-number, so the rows for absorbing states are
        # not valid probability vectors
        policy[absorbing,:] = np.nan
    return policy


def get_unbiased_policy(num_states, num_actions):
    """
    Gets an unbiased policy for given number of states and actions

    parameters
    ----------
    num_states - number of states
    num_actions - number of actions

    returns
    -------
    policy - unbiased policy as (num_states x num_actions) matrix where each
        row has identical elements and sums to 1.
    """
    policy = np.ones((num_states, num_actions))
    policy = policy/policy.sum(axis=1).reshape((num_states,1))
    return policy

def dominant_actions(policy):
    """
    Selects the dominant actions from a policy

    parameters
    ----------
    policy - (num_states x num_actions) matrix of state dependent action
        probabilities.

    returns
    -------
    action_indices - a vector representing a mapping from state to optimal
        action index
    """
    return np.argmax(policy, axis=1)

def indices_to_names(index_sequence, names):
    """
    Maps a sequence of indexes to a sequence of names, e.g. a deterministic
    policy can be mapped from action indexes to action names, or a sequence of
    state indices can be mapped to state names.

    parameters
    ----------
    index_sequence - list or vector of indices referencing items in list names
    names - the corresponding names for the indexes, e.g. action names

    returns
    -------
    name_sequence - a list of names one per item in the input sequence

    """
    names = np.array(names)
    name_sequence = [ names[i] for i in index_sequence ]
    return name_sequence

