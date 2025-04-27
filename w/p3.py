from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
class TDAgent:
    def __init__(self):
        self.gamma = 0.9         # 할인율 (미래 보상 감쇠율)
        self.alpha = 0.1         # 학습률
        self.action_size = 4     # 행동 개수 (상하좌우)
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 정책 π (균등 정책)
        self.V = defaultdict(lambda: 0)                   # 상태가치 함수 V(s)

    def get_action(self, state):
        action_probs = self.pi[state]                 # 현재 상태에서의 정책 π(s)
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)     # 확률적으로 행동 선택

    def eval(self, state, reward, next_state, done):
        next_v = 0 if done else self.V[next_state]    # 종료 상태면 다음 가치 0
        target = reward + self.gamma * next_v         # TD 타깃 값
        self.V[state] += (target - self.V[state]) * self.alpha  # TD 업데이트

env = GridWorld()
agent = TDAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()            # 환경 초기화

    while True:
        action = agent.get_action(state)         # 행동 선택
        next_state, reward, done = env.step(action)  # 환경과 상호작용

        agent.eval(state, reward, next_state, done)  # 가치 함수 업데이트
        if done:
            break
        state = next_state
env.render_v(agent.V)  # 학습된 상태가치 함수 시각화
