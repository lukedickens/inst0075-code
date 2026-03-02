import numpy as np
import copy
import matplotlib.pyplot as plt

from fomlads.rl.environment.states_and_actions import get_unbiased_policy
from fomlads.rl.environment.states_and_actions import get_greedy_policy
from fomlads.rl.environment.states_and_actions import get_epsilon_greedy_policy

def monte_carlo_evaluation(
        env, gamma, policy, num_episodes, max_steps=None, default_value=0):
    """
      Estimates V (state value) function from interacting with an environment
      using the batch method.

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
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
    # initialise returns lists as dictionary of empty lists (one per state)
    # indexed by that state
    returns_lists = { s:[] for s in range(num_states)}
    for _ in range(num_episodes):
        # get a trace by interacting with the environment
        trace = env.run(policy, max_steps=max_steps)
        # iterate over each unique state in the trace and store the return
        # following the first such visit in the corresponding return list
        for s, ret  in trace.first_visit_state_returns(gamma):
            returns_lists[s].append(ret)
    # once all experience is gathered, we take the sample average return from
    # each state as the expected return
    V = default_value * np.ones(num_states)
    for s, returns_list in returns_lists.items():
        # if there are any returns for that state then take the average
        if len(returns_list) > 0:
            V[s] = np.mean(returns_list)
        # otherwise the default value will be used
    # return the value estimates
    return V    

def monte_carlo_q_evaluation(
        env, gamma, policy, num_episodes, max_steps=None, default_value=0):
    """
      Estimates Q (state-action value) function by interacting with an
      environment.

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      policy - (num_states x num_actions)-matrix of policy probabilities
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      Q - an estimate for Q
      
    """
    num_states = env.num_states
    num_actions = env.num_actions
    # initialise returns lists as dictionary of empty lists (one per
    # state-action pair indexed by that pair)
    returns_lists = {
        (s,a):[] for s in range(num_states) for a in range(num_actions)}
    for _ in range(num_episodes):
        # get a trace by interacting with the environment
        trace = env.run(policy, max_steps=max_steps)
        # iterate over unique state-action pairs in the trace and store the
        # return following the first visit in the corresponding return list
        for (s,a), ret  in trace.first_visit_state_action_returns(gamma):
            returns_lists[(s, a)].append(ret)
    # once all experience is gathered, we take the sample average return from
    # each state as the expected return
    Q = default_value * np.ones((num_states, num_actions))
    for (s, a), returns_list in returns_lists.items():
        # if there are any returns for that state-action pair then take the
        # average
        if len(returns_list) > 0:
            Q[s,a] = np.mean(returns_list)
        # otherwise the default value will be used
    # return the value estimates
    return Q

def monte_carlo_batch_optimisation(
        env, gamma, epsilon, num_batches, num_episodes,
        max_steps=None, initial_Q=None, default_value=0):
    """
      Estimates optimal policy based on monte-carlo estimates of the
      Q-function.

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      num_batches - number of batches, where a batch is a monte-carlo estimate
          followed by an epsilon-greedy policy update 
      num_episodes - number of episodes to run per batch
      epsilon - the epsilon to use with epsilon greedy policies 
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)
      initial_Q (optional) - the initial values for the Q-function
      default_value (optional) - if initial values for the Q-function are not
          provided, this is the mean value of the initial Q-function values

      returns
      -------
      policy - (num_states x num_actions)-matrix of policy probabilities for
          estimated optimal policy (will be deterministic so each row will
          have one 1 and the other values will be 0)
      Q - a Q-function estimate of the output policy
    """
    num_states = env.num_states
    num_actions = env.num_actions
    if initial_Q is None:
        # we initialise Q randomly around the default value
        Q = np.random.normal(loc=default_value, size=(num_states, num_actions))
    else:
        # an initial set of Q-values is provided (good for follow on learning)
        Q = initial_Q
    for _ in range(num_batches):
        # the control policy is the epsilon greedy policy from the Q estimates
        control_policy = get_epsilon_greedy_policy(
            epsilon, Q, absorbing=env.absorbing)
        # the updated Q estimates are obtained from monte-carlo Q evaluation
        Q = monte_carlo_q_evaluation(
            env, gamma, control_policy, num_episodes=num_episodes,
            max_steps=max_steps, default_value=default_value)
    return get_greedy_policy(Q, absorbing=env.absorbing), Q

def monte_carlo_iterative_optimisation(
        env, gamma, epsilon, alpha, num_episodes, max_steps=None,
        initial_Q=None, default_value=0):
    """
      Estimates optimal policy based on monte-carlo estimates of the
      Q-function which are approximated using the iterative update method

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - iterative update step size
      num_episodes - number of episodes to run in total
      epsilon - the epsilon to use with epsilon greedy policies 
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)
      initial_Q (optional) - the initial values for the Q-function
      default_value (optional) - if initial values for the Q-function are not
          provided, this is the mean value of the initial Q-function values

      returns
      -------
      policy - (num_states x num_actions)-matrix of policy probabilities for
          estimated optimal policy (will be deterministic so each row will
          have one 1 and the other values will be 0)
      Q - a Q-function estimate of the output policy
    """
    num_states = env.num_states
    num_actions = env.num_actions
    if initial_Q is None:
        # we initialise Q randomly around the default value
        Q = np.random.normal(loc=default_value, size=(num_states, num_actions))
    else:
        # an initial set of Q-values is provided (good for follow on learning)
        Q = initial_Q
    for _ in range(num_episodes):
        # the control policy is the epsilon greedy policy from the Q estimates
        control_policy = get_epsilon_greedy_policy(
            epsilon, Q, absorbing=env.absorbing)
        # get a trace by interacting with the environment
        trace = env.run(control_policy, max_steps=max_steps)
        # iterate over unique state-action pairs in the trace and store the
        # return following the first visit in the corresponding return list
        for (s,a), ret  in trace.first_visit_state_action_returns(gamma):
            Q[s,a] +=  alpha *(ret - Q[s,a])
    return get_greedy_policy(Q, absorbing=env.absorbing), Q

