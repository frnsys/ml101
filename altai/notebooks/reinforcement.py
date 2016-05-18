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
    def __init__(self, state, environment, rewards, discount=0.5, explore=1, learning_rate=0.5, decay=0.005):
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

        # our state is just our position
        self.state = state
        self.env = environment

        # initialize Q
        self.Q = {}

    def actions(self, state):
        return self.env.actions(state)

    def _take_action(self, state, action):
        r, c = state
        if action == 'up':
            r -= 1
        elif action == 'down':
            r += 1
        elif action == 'right':
            c += 1
        elif action == 'left':
            c -= 1

        # return new state
        return (r,c)

    def step(self, action=None):
        """take an action"""
        # check possible actions given state
        actions = self.actions(self.state)

        # if this is the first time in this state,
        # initialize possible actions
        if self.state not in self.Q:
            self.Q[self.state] = {a: 0 for a in actions}

        if action is None:
            if random.random() < self.explore:
                action = random.choice(actions)
            else:
                action = self._best_action(self.state)
        elif action not in actions:
            raise ValueError('unrecognized action!')

        # remember this state and action
        # so we can later remember
        # "from this state, taking this action is this valuable"
        prev_state = self.state

        # decay explore
        self.explore = max(0, self.explore - self.decay)

        # update state
        self.state = self._take_action(self.state, action)

        # update the previous state/action based on what we've learned
        self._learn(prev_state, action, self.state)
        return action

    def _best_action(self, state):
        """choose the best action given a state"""
        actions_rewards = list(self.Q[state].items())
        return max(actions_rewards, key=lambda x: x[1])[0]

    def _learn(self, prev_state, action, new_state):
        """update Q-value for the last taken action"""
        if new_state not in self.Q:
            self.Q[new_state] = {a: 0 for a in self.actions(new_state)}
        self.Q[prev_state][action] = self.Q[prev_state][action] + self.learning_rate * (self.R(new_state) + self.discount * max(self.Q[new_state].values()) - self.Q[prev_state][action])

if __name__ == '__main__':
    interactive = False
    steps = 500
    delay = 0.05

    env = Environment([
        [-10,0,0,50],
        [0,10,100, 0, -100, 20],
        [0,0, None, 10, None, -10, None],
        [None,0, 5, 10, None, 500, 0]
    ])
    pos = random.choice(env.positions)

    # simple reward function
    def reward(state):
        return env.value(state)

    # try discount=0.1 and discount=0.9
    agent = QLearner(pos, env, reward, discount=0.1, learning_rate=0.8, decay=0.5/steps)
    env.render(agent.state)
    for i in range(steps):
        if not interactive:
            agent.step()
        else:
            actions = agent.actions(agent.state)
            action = None
            while action is None:
                try:
                    action = input('what should I do? {} >>> '.format(actions))
                    agent.step(action)
                except ValueError:
                    action = None
        env.render(agent.state)
        print('step: {}, explore: {}, discount: {}'.format(i, agent.explore, agent.discount))
        for pos, vals in agent.Q.items():
            print('{} -> {}'.format(pos, vals))
        time.sleep(delay)
