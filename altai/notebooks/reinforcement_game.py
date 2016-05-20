import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from collections import deque
from lib import Game



class Agent():
    def __init__(self, env, explore=0.1, discount=0.9, hidden_size=100, memory_limit=5000):
        self.env = env
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(env.height * env.width,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(env.n_actions))
        model.compile(loss='mse', optimizer='sgd')
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

        batch_size = min(batch_size, len(self.memory))

        inputs = []
        targets = []

        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size)
        random.shuffle(batch)
        for state, action, next_state, reward in batch:
            inputs.append(state)
            target = self.Q.predict(state)[0]

            # debug, "this should never happen"
            assert not np.array_equal(state, next_state)

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

    def replay(self, batch_size):
        inputs, targets = self._prep_batch(batch_size)
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
    fname = '../data/game_weights.h5'

    if os.path.isfile(fname):
        agent.load(fname)
    else:
        print('training...')
        epochs = 10000
        batch_size = 256

        # keep track of past record_len results
        record_len = 100
        record = deque([], record_len)

        for i in range(epochs):
            game.reset()
            reward = 0
            loss = 0
            # rewards only given at end of game
            while reward == 0:
                prev_state = game.state
                action = agent.choose_action()
                game.move(action)
                reward = game.update()
                new_state = game.state

                # debug, "this should never happen"
                assert not np.array_equal(new_state, prev_state)

                agent.remember(prev_state, action, new_state, reward)
                loss += agent.replay(batch_size)

            sys.stdout.flush()
            sys.stdout.write('epoch: {:04d}/{} | loss: {:.3f} | win rate: {:.3f}\r'.format(i+1, epochs, loss, sum(record)/len(record) if record else 0))

            record.append(reward if reward == 1 else 0)

        agent.save(fname)

    game.reset()
    game.render()
    reward = 0
    while reward == 0:
        action = agent.choose_action()
        game.move(action)
        reward = game.update()
        game.render()
        sleep(0.1)
    print('winner!' if reward == 1 else 'loser!')
