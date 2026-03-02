import numpy as np
import copy
import matplotlib.pyplot as plt

from inst0075.rl.environment.states_and_actions import get_unbiased_policy

def policy_evaluation(
        model, gamma, policy, threshold=1e-8, max_iterations=None):
    """
      Estimates V (state value) function from MDP model with a dynamic
      programming approach

      parameters
      ----------
      model - a model (MDP) of the environment
      gamma - the geometric discount for calculating returns
      threshold - accuracy threshold: an iteration where the maximum change in
          V estimates is below this causes the algorithm to terminate
      max_iterations - maximum number of iterations for the evaluation
      policy - (SxA)-matrix of policy probabilities 
      
      returns
      -------
      V - an estimate for V
      
    """
    assert model.num_states == policy.shape[0]
    assert model.num_actions == policy.shape[1]
    num_states, num_actions = policy.shape
    # shortcuts for reward and transition function
    r = model.r
    t = model.t
    # initialise value estimates
    V = np.zeros(model.num_states)
    updated_V = np.copy(V)

    # ensure initial change is greater than threshold
    change = 2*threshold
    if max_iterations is None:
        # ensure there is no maximum number of iterations
        max_iterations = np.inf
    iteration = 0
    while change >= threshold and iteration < max_iterations:
        iteration += 1
        # iterate over state indices, checking if each is an absorbing state
        ## Notation:
        ##      s = prior state index
        ##      a = prior action index
        ##      s_ = posterior state index
        for s, is_absorbing in enumerate(model.absorbing):
            # do not update absorbing states
            if is_absorbing: 
                continue
            # for non-absorbing states 
            # tmp_V will collect the state value for s
            tmp_V = 0
            for a in range(model.num_actions):
                # tmp_Q will collect the state-action value for (s,a)
                tmp_Q = 0
                for s_ in range(model.num_states):
                    tmp_Q += t(s,a,s_)*(r(s,a,s_) + gamma*V[s_])
                # add state-action value to state value
                tmp_V += policy[s,a]*tmp_Q
            # store the state value for s
            updated_V[s] = tmp_V
        # determine the maximum change in any V[s]
        change = np.max(abs(updated_V - V))
        # store the new values
        V = np.copy(updated_V)
    return V    


def policy_iteration(
        model, gamma, threshold=1e-8, max_iterations=None):
    """
      Estimates V (state value) function from MDP model with a dynamic
      programming approach

      parameters
      ----------
      model - a model (MDP) of the environment
      gamma - the geometric discount for calculating returns
      threshold - accuracy threshold for policy evaluation an iteration where
          the maximum change in V estimates is below this causes the policy 
          evaluation algorithm to terminate
      max_iterations - maximum number of iterations for the evaluation
      
      returns
      -------
      policy - a deterministic policy in terms of an (SxA)-matrix of policy
          probabilities 
      
    """
    # short names for number of states and actions
    num_states = model.num_states
    num_actions = model.num_actions
    # shortcuts for reward and transition function
    r = model.r
    t = model.t
    # initialise policy as unbiased random policy
    policy = get_unbiased_policy(num_states, num_actions)
    # initialise stable
    stable = False
    while not stable:
        V = policy_evaluation(
            model, gamma, policy, threshold=threshold,
            max_iterations=max_iterations)
        stable = True
        for s, is_absorbing in enumerate(model.absorbing):
            # old best action is the one with highest probability in the 
            # current policy (the 1st index with equal probabilities is chosen)
            best_action = np.argmax(policy[s,:])
            new_Q = np.zeros(num_actions)
            for a in range(num_actions):
                for s_ in range(num_states):
                    new_Q[a] += t(s,a,s_)*(r(s,a,s_) + gamma*V[s_])
            new_best_action = np.argmax(new_Q)
            if new_best_action != best_action:
                stable = False
            # update the policy to reflect the new best action for s
            policy[s,:] = 0
            policy[s,new_best_action] = 1
    return policy

