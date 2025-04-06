from collections import defaultdict  # 기본값을 제공하는 딕셔너리 사용
import numpy as np  # 수학 계산 라이브러리 NumPy 임포트
from gridworld import GridWorld  # GridWorld 환경 임포트


# 랜덤 에이전트 클래스 정의 (몬테카를로 평가)
class RandomAgent:
    def __init__(self):
        self.gamma = 0.9  # 감마: 미래 보상 할인율
        self.action_size = 4  # 가능한 행동의 개수

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 초기 랜덤 정책 설정
        self.pi = defaultdict(lambda: random_actions)  # 정책 저장소 초기화
        self.V = defaultdict(lambda: 0)  # 상태 가치 함수 저장소 초기화
        self.cnts = defaultdict(lambda: 0)  # 상태 방문 횟수 저장소 초기화
        self.memory = []  # 에피소드 데이터를 저장할 메모리

    def get_action(self, state):
        action_probs = self.pi[state]  # 현재 상태의 행동 확률 가져오기
        actions = list(action_probs.keys())  # 가능한 행동 리스트 추출
        probs = list(action_probs.values())  # 각 행동의 확률 추출
        return np.random.choice(actions, p=probs)  # 확률에 따라 랜덤하게 행동 선택

    def add(self, state, action, reward):
        data = (state, action, reward)  # 상태-행동-보상 데이터 생성
        self.memory.append(data)  # 메모리에 데이터 추가

    def reset(self):
        self.memory.clear()  # 메모리 초기화

    def eval(self):
        G = 0  # 누적 보상 초기화
        for data in reversed(self.memory):  # 에피소드 데이터를 역순으로 처리
            state, action, reward = data
            G = self.gamma * G + reward  # 누적 보상 계산
            self.cnts[state] += 1  # 해당 상태 방문 횟수 증가
            self.V[state] += (G - self.V[state]) / self.cnts[state]  # 상태 가치 함수 갱신


# 환경 및 에이전트 초기화
env = GridWorld()  # GridWorld 환경 생성
agent = RandomAgent()  # 랜덤 에이전트 생성

# 에피소드 실행 루프
episodes = 1000
for episode in range(episodes):
    state = env.reset()  # 환경 초기화 및 시작 상태 설정
    agent.reset()  # 에이전트 메모리 초기화

    while True:
        action = agent.get_action(state)  # 현재 상태에서 행동 선택
        next_state, reward, done = env.step(action)  # 선택한 행동 수행 후 결과 얻기

        agent.add(state, action, reward)  # 메모리에 상태-행동-보상 데이터 추가
        if done:
            agent.eval()  # 에피소드 종료 시 가치 함수 갱신 (몬테카를로 방법 사용)
            break

        state = next_state

    # 몬테카를로 방법으로 얻은 가치 함수 시각화
env.render_v(agent.V)