import numpy as np

from fomlads.rl.environment.simulation import Simulation

class SimpleQueuingSim(Simulation):
    """
    A simulation of the simple queuing problem from Sutton and Barto, example 6.7

    must define reset(), next(action), is_terminal() and step(action).
    See MDPSimulation for examples of these.
    """
    def __init__(self, num_servers, p, queue_length):
        """
        parameters
        ----------
        num_servers - number of servers to which we can assign jobs
        p - probability at each time-step that a busy server will free up
        queue_length - number of jobs in the queue
        """
        self.num_servers = num_servers
        self.p = p
        # queue is an array of priority indices 0, 1, 2, 3
        # (priority values are 1,2,4,8 from Sutton)
        # we don't actually need to model the queue itself for Sutton and
        # Barto's example but I have included it to help you think about
        # extensions
        self.queue_length = queue_length
        self.priority_values = np.array([1,2,4,8])
        self.priority_probs = np.array([0.5/3, 0.5/3, 0.5/3, 0.5])
        # state_names are the observable state names (not the hidden state
        # which corresponds to the priority of each job in the queue)
        # this is useful if applying table lookup approaches
        state_names = []
        for servers_busy in range(self.num_servers+1):
            for priority in self.priority_values:
                state_names.append((servers_busy, priority))
        num_states = len(state_names)
        self.absorbing = np.zeros(num_states, dtype=bool)
        # we would like to be able to lookup the state index from the
        # (observable) state description (servers_busy, priority)
        self.state_lookup = {}
        for i, (servers_busy, priority) in enumerate(state_names):
            self.state_lookup[(servers_busy, priority)] = i
        #
        action_names = ['accept', 'reject']
        num_actions = len(action_names)
        # now call the __init__ function on the superclass
        super().__init__(num_states, num_actions, state_names, action_names) 


    def reset(self):
        # reset the step and total reward counters in Simulation
        self.reset_counts()
        self.queue = np.random.choice(
            self.priority_values, size=self.queue_length, p=self.priority_probs)
        # servers are either busy or free, if busy they become free with
        # probability self.p
        self.servers_busy = 0
        rep = self.raw_state_representation()
        return self.state_lookup[rep]

    def next(self, action):
        # see how many servers will free up.
        newly_free = np.random.binomial(self.servers_busy, self.p)
        self.servers_busy -= newly_free
        if action == 0:
            # accept job
            if self.servers_busy < self.num_servers:
                reward = self.queue[0]
                self.servers_busy += 1
            else:
                # cannot assign to server, so penalise
                reward = -self.queue[0]
        else:
            # reject job
            reward = 0
        # move jobs forward in queue and generate next job in last position
        self.queue[:-1] = self.queue[1:]            
        self.queue[-1] = np.random.choice(
            self.priority_values, p=self.priority_probs)
        # new observed state depends on the number of busy servers and 
        rep = self.raw_state_representation()
        next_state = self.state_lookup[rep]
        # update total reward and steps for this episode
        self.increment_counts(reward)
        # 
        return next_state, reward

    def raw_state_representation(self):
        return (self.servers_busy, self.queue[0])

    def is_terminal(self):
        return False


class SimpleQueuingSimFeatures(SimpleQueuingSim):
    def __init__(self, num_servers, p, queue_length):
        """
        parameters
        ----------
        num_servers - number of servers to which we can assign jobs
        p - probability at each time-step that a server will free up
        queue_length - number of jobs in the queue
        """
        # initilise via the superclass
        super().__init__(num_servers, p, queue_length)

    def reset(self):
        # initialise the state
        _ = super().reset()
        # but instead of returning the state, we return the representation
        return self.state_representation()

    def next(self, action):
        # use the superclass next function to evolve the system
        next_state, reward = super().next(action)
        # states are now vectors of features
        features = self.state_representation()
        return features, reward

    def state_representation(self):
        # shorthand for readability
        q = self.queue
        pv = self.priority_values
        num_priorities = pv.shape[0]
        # feature is made of three subparts part1, part2 and part3
        # part1 - we want to capture the number of servers that are free
        # for simplicity we will capture this as a simple number
        features1 = [self.servers_busy]
        # part2 - a one hot vector of the next priority in the queue
        # create zero vector then set to one the ith element corresponding to
        # the index of the priority of the first job
        features2 = np.zeros(num_priorities)
        features2[q[0]==pv] = 1
        # part3 - a feature of counts of each priority in the queue
        features3 = np.sum(
            q.reshape((-1,1)) == pv.reshape((1,-1)), axis=0)
        features = np.concatenate((features1, features2, features3))
        return features

