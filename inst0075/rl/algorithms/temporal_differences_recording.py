import numpy as np
import copy
import matplotlib.pyplot as plt

from inst0075.rl.environment.states_and_actions import choose_from_policy

from inst0075.rl.environment.states_and_actions import get_unbiased_policy
from inst0075.rl.environment.states_and_actions import get_greedy_policy
from inst0075.rl.environment.states_and_actions import get_epsilon_greedy_policy

def temporal_difference_evaluation(
        env, gamma, alpha, policy, num_episodes, max_steps=np.inf,
        initial_V=None):
    """
      Estimates V (state value) function from interacting with an environment.

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      policy - (num_states x num_actions)-matrix of policy probabilities
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      V - an estimate for V
      
    """
    num_states = env.num_states
    num_actions = env.num_actions
    td_errors = {}
    if initial_V is None:
        V = np.zeros(num_states)
    else:
        V = initial_V
    for episode in range(num_episodes):
        # initialise state
        s = env.reset()
        steps = 0
        while not env.is_terminal() and steps < max_steps:
            a = choose_from_policy(policy, s)
            next_s, r = env.next(a)
            # get td_error (called delta on slides)
            td_error = r + gamma*V[next_s] - V[s]
            these_errors = td_errors.get(s,[])
            these_errors.append((episode, steps, td_error))
            td_errors[s] = these_errors
            # update the value function estimate
            V[s] += alpha*td_error
            # set next state to current state
            s = next_s
            # increment the number of steps
            steps += 1
    # return the value estimates
    return V, td_errors

def sarsa(
        env, gamma, alpha, epsilon, num_episodes, max_steps=np.inf,
        initial_Q=None):
    """
      Estimates optimal policy by interacting with an environment using
      a td-learning approach

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      policy - an estimate for the optimal policy
      Q - a Q-function estimate of the output policy
    """
    num_states = env.num_states
    num_actions = env.num_actions
    td_errors = {}
    if initial_Q is None:
        Q = np.zeros((num_states, num_actions))
    else:
        Q = initial_Q
    policy = get_epsilon_greedy_policy(epsilon, Q, env.absorbing)
    for _ in range(num_episodes):
        # initialise state
        s = env.reset()
        # choose initial action
        a = choose_from_policy(policy, s)

        steps = 0
        while not env.is_terminal() and steps < max_steps:
            next_s, r = env.next(a)
            # choose the next action
            next_a = choose_from_policy(policy, s)
            # get td_error (called delta on slides)
            td_error = r + gamma*Q[next_s, next_a] - Q[s, a]
            these_errors = td_errors.get((s,a),[])
            these_errors.append(td_error)
            td_errors[(s,a)] = these_errors
            # update the Q function estimate
            Q[s,a] += alpha*td_error
            # update the policy (only need to do so for current state)
            policy[s,:] = get_epsilon_greedy_policy(
                epsilon, Q[s,:].reshape((1,num_actions)))
            # set next state and action to current state and action
            s = next_s
            a = next_a
            # increment the number of steps
            steps += 1
    # return the policy
    return policy, Q, td_errors

def q_learning(
        env, gamma, alpha, epsilon, num_episodes, max_steps=np.inf,
        initial_Q=None):
    """
      Estimates optimal policy by interacting with an environment using
      a td-learning approach

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      policy - an estimate for the optimal policy
      Q - a Q-function estimate of the optimal epsilon-soft policy
    """
    num_states = env.num_states
    num_actions = env.num_actions
    td_errors = {}
    if initial_Q is None:
        Q = np.zeros((num_states, num_actions))
    else:
        Q = initial_Q
    policy = get_epsilon_greedy_policy(epsilon, Q, env.absorbing)
    for _ in range(num_episodes):
        # initialise state
        s = env.reset()
        steps = 0
        while not env.is_terminal() and steps < max_steps:
            # choose the action
            a = choose_from_policy(policy, s)
            next_s, r = env.next(a)
            # get td_error (called delta on slides)
            td_error = r + gamma*np.max(Q[next_s, :]) - Q[s, a]
            these_errors = td_errors.get((s,a),[])
            these_errors.append(td_error)
            td_errors[(s,a)] = these_errors
            # update the Q function estimate
            Q[s,a] += alpha*td_error
            # update the policy (only need to do so for current state)
            policy[s,:] = get_epsilon_greedy_policy(
                epsilon, Q[s,:].reshape((1,num_actions)))
            # set next state and action to current state and action
            s = next_s
            # increment the number of steps
            steps += 1
    # return the policy
    return get_greedy_policy(Q, absorbing=env.absorbing), Q, td_errors

