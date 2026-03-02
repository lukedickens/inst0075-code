import math
import numpy as np


class MDP(object):
    """
    Defines a Markov Decision process, consisting of states, actions,
    a transition function (or matrix) and a reward function (or matrix). It
    also specifies which states are absorbing and the distribution over initial
    states.
    """
    def __init__(
            self, states, actions, initial, absorbing, T_matrix=None,
            R_matrix=None, t_function=None, r_function=None):
        """
        Constructs an MDP

        parameters
        ----------
        states - either a list of the state names or the number of states
        actions - either a list of the action names or the number of as
        initial - an initial state probability vector over all states
        absorbing - a boolean vector over all states indicating which are
            absorbing. ith elem True if ith state absorbing
        [optional]
        T_matrix - encodes transition probs in |A|x|S|x|S|-matrix,
            where T_matrix[a,s0,s1] is the prob t(s0,a,s1)
        R_matrix - encodes rewards |A|x|S|x|S|-matrix, where
            R_matrix[a,s0,s1] is the reward for transition (s0, a, s1)
        t_function - encodes transition probs as a function converted
            to transition matrix on construction (can only defin a transition
            function or transition matrix not both)
        r_function - encodes rewards in a function (can only define a reward
            function or reward matrix not both)
        """
        # resolve number and names of states and as
        if type(states) is list:
            self.state_names = states
            self.num_states = len(self.state_names)
        else:
            self.num_states = states
            self.state_names = self.list_of_names('s', self.num_states)
        if type(actions) is list:
            self.action_names = actions
            self.num_actions = len(self.action_names)
        else:
            self.num_actions = actions
            self.action_names = self.list_of_names('a', self.num_actions)
        # transitions and rewards can be defined as functions or matrices
        if not T_matrix is None:
            self.T = T_matrix
        else:
            self.T = self.convert_function_to_matrix(
                t_function, self.num_states, self.num_actions)
        if not R_matrix is None:
            self.R = R_matrix
        else:
            self.R = self.convert_function_to_matrix(
                r_function, self.num_states, self.num_actions)
        # simply store other components
        self.initial = initial
        self.absorbing = absorbing

    def r(self, s, a, s_):
        """
        Get reward for given transition

        parameters
        ----------
        s - the prior state index
        a - the action index
        s_ - the posterior state index

        returns
        ------
        reward value for given transition
        """
        return self.R[a, s, s_]

    def t(self, s, a, s_=None):
        """
        Get transition distribution or probability

        parameters
        ----------
        s - the prior state index
        a - the action index
        s_ (optional) - the posterior state index

        returns
        ------
        Either the transition distribution (as probability vector) over
        subsequent states or the probability of transitioning to a given state.
        """
        # if s_ is None, return distribution over all subsequent states
        if s_ is None:
            return self.T[a, s, :]
        # else return single probability
        return self.T[a, s, s_]

    def list_of_names(self, base_name, n):
        """
        A helper method that converts a base name and number of elements n
        into a list of names
        """
        # number of padded zeros
        num_digits = np.ceil(np.log10(n-1))
        fmt = base_name + '%0'+str(num_digits)+'d'
        return [ fmt % i for i in range(n) ]

    @classmethod
    def convert_function_to_matrix(cls, func, num_states, num_actions):
        matrix = np.empty((num_actions, num_states, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                for s_ in range(num_states):
                    matrix[a,s,s_] = func(s,a,s_)
        return matrix

    def pretty_print_T(self):
        """
        Gets the human readable version of the transition matrix
        """
        s = ""
        for a, action in enumerate(self.action_names):
            s += "transition probs for: " + action +"\n\n"
            s += self.pretty_print_2d_matrix(
                self.T[a,:,:], self.state_names, self.state_names)    
            s += "\n"
        return s
    
    @classmethod
    def pretty_print_2d_matrix(cls, matrix, row_labels, column_labels):
        r_lab_width = max(len(r_lab) for r_lab in row_labels)
        c_lab_width = max(len(c_lab) for c_lab in column_labels)
        r_lab_tabs = math.ceil((r_lab_width+1.)/8)
        c_lab_tabs = math.ceil((c_lab_width+1.)/8)
        # header row starts with blank
        s = "\t"*r_lab_tabs
        # ...followed by column labels
        for c_lab in column_labels:
            s += cls.tab_pad(c_lab, c_lab_tabs)
        s += "\n"
        # each subsequent row has row label followed by row elements
        # appropriately tab padded
        for r_lab, row in zip(row_labels, matrix):
            s += cls.tab_pad(r_lab, r_lab_tabs)
            for elem in row:
                s += cls.tab_pad(str(elem), c_lab_tabs)
            s += "\n"
        return s

    @classmethod
    def tab_pad(cls, text, num_tabs, tab_width=8):
        return text + "\t"*(num_tabs-len(text)//tab_width)

