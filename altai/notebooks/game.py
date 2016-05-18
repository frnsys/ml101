import random
import numpy as np
from blessings import Terminal
from keras.models import Sequential
from keras.layers.core import Dense
from collections import deque


class Game():
    def __init__(self, shape=(10,10)):
        self.shape = shape
        self.height, self.width = shape
        self.last_row = self.height - 1
        self.n_actions = 3 # left, right, stay
        self.term = Terminal()
        self.reset()

    def reset(self):
        # black & white only
        self.grid = np.zeros(self.shape, dtype=np.bool)

        # can only move left or right (or stay)
        # so position is only its horizontal position (col)
        self.pos = np.random.randint(self.width - 1)

        # start in the bottom-left corner
        self.set_position((self.last_row, self.pos), 1)

        # item to catch
        self.target = (0, np.random.randint(self.width - 1))
        self.set_position(self.target, 1)

    def move(self, action):
        # clear previous space
        self.set_position((self.last_row, self.pos), 0)

        # action is either -1, 0, 1,
        # but comes in as 0, 1, 2, so subtract 1
        action -= 1
        self.pos = min(max(self.pos + action, 0), self.width - 1)

        # set new position
        self.set_position((self.last_row, self.pos), 1)

    @property
    def state(self):
        # state only needs to consist of
        # the target's row & col and the player's col
        # return np.array([self.target[0], self.target[1], self.pos])
        return self.grid.reshape((1,-1))

    def set_position(self, pos, val):
        r, c = pos
        self.grid[r,c] = val

    def update(self):
        r, c = self.target

        # off the map, it's gone
        if r == self.last_row:
            # reward of 1 if collided with player, else -1
            return 1 if c == self.pos else -1

        self.set_position(self.target, 0)
        self.target = (r+1, c)
        self.set_position(self.target, 1)
        return 0

    def render(self):
        print(self.term.clear())
        for r, row in enumerate(self.grid):
            for c, on in enumerate(row):
                if on:
                    color = 235
                else:
                    color = 229

                print(self.term.move(r, c) + self.term.on_color(color) + ' ' + self.term.normal)

        # move cursor to end
        print(self.term.move(self.height, 0))


class Agent():
    def __init__(self, env, explore=0.2, discount=0.9, hidden_size=100, memory_limit=500):
        self.env = env
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(env.height * env.width,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(env.n_actions))
        model.compile(loss='mse', optimizer='rmsprop')
        self.Q = model

        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)

        self.explore = explore
        self.discount = discount

    def choose_action(self):
        if np.random.rand() <= self.explore:
            return np.random.randint(0, self.env.n_actions)
        state = self.env.state
        q = self.Q.predict(state)
        return np.argmax(q[0])

    def remember(self, state, action, next_state, reward):
        # the deque object will automatically keep a fixed length
        self.memory.append((state, action, next_state, reward))

    def _prep_batch(self, batch_size):
        if batch_size > self.memory.maxlen:
            Warning('batch size should not be larger than max memory size. Setting batch size to memory size')
            batch_size = self.memory.maxlen

        # only start replaying once we have enough memories
        if len(self.memory) < batch_size:
            return

        inputs = []
        targets = []

        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size)
        for state, action, next_state, reward in batch:
            inputs.append(state)
            target = self.Q.predict(state)[0]

            # non-zero reward indicates terminal state
            if reward:
                target[action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                Q_sa = np.max(self.Q.predict(next_state)[0])
                target[action] = reward + self.discount * Q_sa
            targets.append(target)

        # to numpy matrices
        return np.vstack(inputs), np.vstack(targets)

    def replay(self, batch_size=10):
        batch = self._prep_batch(batch_size)
        if batch is None:
            return 0
        inputs, targets = batch
        loss = self.Q.train_on_batch(inputs, targets)
        return loss

    def save(self, fname):
        self.Q.save_weights(fname)

    def load(self, fname):
        self.Q.load_weights(fname)


if __name__ == '__main__':
    import os
    import sys
    from time import sleep
    game = Game()
    agent = Agent(game)

    wins = 0
    fname = 'game_weights.h5'

    if os.path.isfile(fname):
        agent.load(fname)
    else:
        print('training...')
        epochs = 1000
        batch_size = 50

        for i in range(epochs):
            game.reset()
            reward = 0
            loss = 0
            state = game.state
            # rewards only given at end of game
            while reward == 0:
                action = agent.choose_action()
                game.move(action)
                reward = game.update()
                new_state = game.state

                agent.remember(state, action, new_state, reward)
                loss += agent.replay(batch_size)
                state = new_state

            sys.stdout.write('epoch: {}/{} | loss: {:.3f} | win rate: {:.3f}\r'.format(i+1, epochs, loss, wins/(i+1)))
            sys.stdout.flush()

            wins += reward if reward == 1 else 0

        agent.save('game_weights.h5')

    game.reset()
    while reward == 0:
        game.render()
        action = agent.choose_action()
        game.move(action)
        reward = game.update()
        sleep(0.1)
    print('winner!' if reward == 1 else 'loser!')
