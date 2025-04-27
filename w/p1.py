from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld

class TDAgent:
    def __init__(self):
        self.V = defaultdict(float)
        self.gamma = 0.9
        self.alpha = 0.8

    def update(self, state, reward, next_state, done):
        next_v = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_v
        self.V[state] += (target - self.V[state]) * self.alpha

env = GridWorld()
agent = TDAgent()
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    while True:
        action = np.random.choice(4)
        next_state, reward, done = env.step(action)
        agent.update(state, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_v(agent.V)
