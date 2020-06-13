import abc

from rlkit.policies.base import ExplorationPolicy


class ExplorationStrategy(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action(self, t, observation, policy, **kwargs):
        pass

    def reset(self):
        pass


class RawExplorationStrategy(ExplorationStrategy, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_action_from_raw_action(self, action, **kwargs):
        pass

    def get_action(self, t, policy, *args, **kwargs):
        action, agent_info = policy.get_action(*args, **kwargs)
        return self.get_action_from_raw_action(action, t=t), agent_info

    def reset(self):
        pass


class PolicyWrappedWithExplorationStrategy(ExplorationPolicy):
    def __init__(
            self,
            exploration_strategy: ExplorationStrategy,
            policy,
    ):
        self.es = exploration_strategy
        self.policy = policy
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def get_action(self, *args, **kwargs):
        return self.es.get_action(self.t, self.policy, *args, **kwargs)

    def reset(self):
        self.es.reset()
        self.policy.reset()

import random
import numpy as np

class PushPrimitiveWithExplorationStrategy(ExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """
    def __init__(
            self,
            policy,
            primitive_policy,
            init_prob = 0.5,
            decay_rate = 1, # deflaut no decay


    ):

        self.policy = policy
        self.t = 0
        self.prob_random_action = init_prob
        self.primitive_policy = primitive_policy
        self.init_prob = init_prob
        self.decay_rate = decay_rate

    def get_action(self,  *args, **kwargs):
        if random.random() <= self.prob_random_action:
            action, agent_info = self.primitive_policy.get_action(*args, **kwargs)
        else:

            action, agent_info = self.policy.get_action(*args, **kwargs)
        return action, agent_info

    def reset(self):

        self.primitive_policy.reset()
        self.policy.reset()

    def update_explor_rate(self, epoch):

        self.prob_random_action = max(self.init_prob * np.power(self.decay_rate, epoch),0.1) #if self.decay_rate!=1 else self.prob_random_action

        print('exploration rate :', self.prob_random_action)

