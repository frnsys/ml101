import random
from lib import Environment

class QLearner():
    def __init__(self, state, environment, rewards, discount=0.5, explore=0.5, learning_rate=1):
        """
        - state: the agent's starting state
        - rewards: a reward function, taking a state as input, or a mapping of states to a reward value
        - discount: how much the agent values future rewards over immediate rewards
        - explore: with what probability the agent "explores", i.e. chooses a random action
        - learning_rate: how quickly the agent learns. For deterministic environments (like ours), this should be left at 1
        """
        self.discount = discount
        self.explore = explore
        self.learning_rate = learning_rate
        self.R = rewards.get if isinstance(rewards, dict) else rewards

        # our state is just our position
        self.state = state
        self.reward = 0
        self.env = environment

        # initialize Q
        self.Q = {}

    def reset(self, state):
        self.state = state
        self.reward = 0

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
        reward = self.R(new_state)
        self.reward += reward
        self.Q[prev_state][action] = self.Q[prev_state][action] + self.learning_rate * (reward + self.discount * max(self.Q[new_state].values()) - self.Q[prev_state][action])


def choose_action(agent):
    """interactively choose action for agent"""
    actions = agent.actions(agent.state)
    action = None
    while action is None:
        try:
            action = input('what should I do? {} >>> '.format(actions))
            if action == 'quit':
                return True
            agent.step(action)
        except ValueError:
            action = None
    return False


if __name__ == '__main__':
    # define the gridworld environment
    env = Environment([
        [-10,0,0,50],
        [0,10,100, 0, -100, 500],
        [0,0, None, 10, None, -10, None],
        [None,0, 5, 10, None, 5, 0]
    ])

    # start at a random position
    pos = random.choice(env.positions)

    # simple reward function
    def reward(state):
        return env.value(state)

    # try discount=0.1 and discount=0.9
    DISCOUNT = 0.1
    LEARNING_RATE = 1
    agent = QLearner(pos, env, reward, discount=DISCOUNT, learning_rate=LEARNING_RATE)
    env.render(agent.state)

    i = 0
    while True:
        done = choose_action(agent)
        if done:
            break
        env.render(agent.state)
        print('step: {:03d}, explore: {:.2f}, discount: {}'.format(i, agent.explore, agent.discount))
        for pos, vals in agent.Q.items():
            print('{} -> {}'.format(pos, vals))
        i += 1
