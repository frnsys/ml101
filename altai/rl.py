"""
# Reinforcement Learning

Reinforcement learning is a form of machine learning which is great for autonomous agents. Reinforcement learning agents interact with and learn from their environment. Such agents learn how to make rational decisions under conditions of uncertainty.

AlphaGo uses reinforcement learning in combination with deep neural networks to make decisions on what plays to make.

## Q-Learning

The basic form of reinforcement learning is Q-learning.

Given an environment where actions result in uncertain states, Q-learning allows the agent to learn a _policy_ (that is, a function which chooses an action to state given a state).
"""

import time
import random
from lib import Environment


class QLearner():
    def __init__(self, position, environment, rewards, discount=0.5, explore=1, learning_rate=0.5, decay=0.005):
        """
        - states_actions: a mapping of states to viable actions for that state
        - rewards: a reward function, taking a state as input, or a mapping of states to a reward value
        - discount: how much the agent values future rewards over immediate rewards
        - explore: with what probability the agent "explores", i.e. chooses a random action
        - decay: how much to decay the explore rate with each step
        - learning_rate: how quickly the agent learns
        """
        self.discount = discount
        self.explore = explore
        self.decay = decay
        self.learning_rate = learning_rate
        self.R = rewards.get if isinstance(rewards, dict) else rewards

        # previous (state, action)
        self.prev = (None, None)

        # our state is just our position
        self.state = position
        self.env = environment

        # initialize Q
        self.Q = {}

    def step(self):
        """take an action"""
        # check possible actions given state
        actions = self.env.actions(self.state)

        # if this is the first time in this state,
        # initialize possible actions
        if self.state not in self.Q:
            self.Q[self.state] = {a: 0 for a in actions}

        if random.random() < self.explore:
            action = random.choice(actions)
        else:
            action = self._best_action(self.state)

        # learn from the previous action, if there was one
        self._learn(self.state)

        # remember this state and action
        self.prev = (self.state, action)

        # decay explore
        self.explore = max(0, self.explore - self.decay)

        r, c = self.state
        if action == 'up':
            r -= 1
        elif action == 'down':
            r += 1
        elif action == 'right':
            c += 1
        elif action == 'left':
            c -= 1
        self.state = (r,c)
        return action

    def _best_action(self, state):
        """choose the best action given a state"""
        actions_rewards = list(self.Q[state].items())
        return max(actions_rewards, key=lambda x: x[1])[0]

    def _learn(self, state):
        """update Q-value for the last taken action"""
        p_state, p_action = self.prev
        if p_state is None:
            return
        self.Q[p_state][p_action] = self.learning_rate * (self.R(state) + self.discount * max(self.Q[state].values())) - self.Q[p_state][p_action]


if __name__ == '__main__':
    env = Environment([
        [None, -10,0,0,50],
        [10,0,10,100, 0, -200, 20],
        [10,0,0, None, 10, None, -10, None],
        [-10,None,0, 5, 10, None, 1000, 0]
    ])
    pos = random.choice(env.positions)

    # simple reward function
    def reward(state):
        # add 1 so that the agent is
        # encouraged to explore 0-valued spaces
        return env.value(state) + 1

    agent = QLearner(pos, env, reward)
    for i in range(100):
        agent.step()
        env.render(agent.state)
        print('step: {}, explore: {}'.format(i, agent.explore))
        for pos, vals in agent.Q.items():
            print('{} -> {}'.format(pos, vals))
        time.sleep(0.3)
