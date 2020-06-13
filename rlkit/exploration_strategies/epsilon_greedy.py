import random

from rlkit.exploration_strategies.base import RawExplorationStrategy
import numpy as np

from rlkit.exploration_strategies.base import ExplorationStrategy

class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
        return action

class PushNetEpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            angle = np.random.randint(0,self.action_space[0])
            pix_v = np.random.randint(0,self.action_space[1])
            pix_u = np.random.randint(0, self.action_space[2])

            action = np.array([angle, pix_v, pix_u])

        return action


class PushNetPrimitiveEpsilonGreedy(ExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(self, primitive_policy, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.primitive_policy = primitive_policy

    def get_action(self, t, policy, *args, **kwargs):
        if random.random() <= self.prob_random_action:
            action, agent_info = self.primitive_policy.get_action(**kwargs)
        else:

            action, agent_info = policy.get_action(*args, **kwargs)
        return action, agent_info

