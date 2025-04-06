import numpy as np  # NumPy 라이브러리 임포트 (수학 계산용)
from collections import defaultdict  # 기본값을 제공하는 딕셔너리 사용
from common.gridworld import GridWorld  # GridWorld 환경 임포트 (현재 에러 발생 부분)

# 행동 확률 계산 함수: 탐욕 정책에 따라 행동 확률을 반환
def greedy_probs(Q, state, epsilon=0.1, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]  # 현재 상태에서 각 행동의 Q값 가져오기
    max_action = np.argmax(qs)  # 가장 높은 Q값을 가지는 행동 선택

    base_prob = epsilon / action_size  # 모든 행동에 균등하게 분배할 확률
    action_probs = {action: base_prob for action in range(action_size)}  # 기본 확률 설정
    action_probs[max_action] += (1 - epsilon)  # 가장 높은 Q값을 가지는 행동에 추가 확률 부여
    return action_probs

# 몬테카를로 에이전트 클래스 정의
class McAgent:
    def __init__(self):
        self.gamma = 0.9  # 감마: 미래 보상의 할인율
        self.epsilon = 0.1  # 엡실론: 탐욕 정책의 랜덤성 비율
        self.alpha = 0.1  # 알파: 학습률
        self.action_size = 4  # 가능한 행동의 개수

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 초기 랜덤 정책 설정
        self.pi = defaultdict(lambda: random_actions)  # 정책 저장소 초기화
        self.Q = defaultdict(lambda: 0)  # Q값 저장소 초기화
        self.memory = []  # 에피소드 데이터를 저장할 메모리

    def get_action(self, state):
        action_probs = self.pi[state]  # 현재 상태의 행동 확률 가져오기
        actions = list(action_probs.keys())  # 가능한 행동 리스트 추출
        probs = list(action_probs.values())  # 각 행동의 확률 추출
        return np.random.choice(actions, p=probs)  # 확률에 따라 랜덤하게 행동 선택

    def add(self, state, action, reward):
        data = (state, action, reward)  # 상태, 행동, 보상 데이터 생성
        self.memory.append(data)  # 메모리에 데이터 추가

    def reset(self):
        self.memory.clear()  # 메모리 초기화

    def update(self):
        G = 0  # 누적 보상 초기화
        for data in reversed(self.memory):  # 에피소드 데이터를 역순으로 처리
            state, action, reward = data
            G = self.gamma * G + reward  # 누적 보상 계산
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha  # Q값 업데이트
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon, self.action_size)  # 탐욕 정책 갱신

# 환경 및 에이전트 초기화
env = GridWorld()  # GridWorld 환경 생성 (현재 에러 발생 부분)
agent = McAgent()  # 몬테카를로 에이전트 생성

# 에피소드 실행 루프
episodes = 10000
for episode in range(episodes):
    state = env.reset()  # 환경 초기화 및 시작 상태 설정
    agent.reset()  # 에이전트 메모리 초기화

    while True:
        action = agent.get_action(state)  # 현재 상태에서 행동 선택
        next_state, reward, done = env.step(action)  # 선택한 행동 수행 후 결과 얻기

        agent.add(state, action, reward)  # 메모리에 상태-행동-보상 데이터 추가
        if done:
            agent.update()  # 에피소드 종료 시 정책 및 Q값 업데이트
            break

        state = next_state

# 결과 렌더링 (Q값 시각화)
env.render_q(agent.Q)