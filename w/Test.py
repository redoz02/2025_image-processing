import numpy as np
import random
import pandas as pd

# 보드 크기 설정
BOARD_WIDTH = 6
BOARD_HEIGHT = 6

# 간단한 테트리스 환경 정의
class SimpleTetrisEnv:
    def __init__(self):
        self.reset()

    # 환경 초기화
    def reset(self):
        self.board = np.zeros(BOARD_WIDTH, dtype=int)  # 각 열의 높이 0으로 시작
        return self._get_state()

    # 현재 상태(각 열의 높이)를 튜플로 반환
    def _get_state(self):
        return tuple(self.board)

    # 주어진 행동(action)을 수행하고 다음 상태, 보상, 게임 종료 여부 반환
    def step(self, action):
        if self.board[action] >= BOARD_HEIGHT:
            return self._get_state(), -1, True  # 해당 열이 가득 찼으면 게임 오버
        self.board[action] += 1

        reward = 0
        # 모든 열의 높이가 같으면 보드 초기화 (줄 제거 보상)
        if np.all(self.board == self.board[0]):
            reward = 1
            self.board = np.zeros(BOARD_WIDTH, dtype=int)

        return self._get_state(), reward, False

# Q-learning 설정
num_episodes = 5000  # 학습 횟수
epsilon = 0.1        # 탐험 확률
alpha = 0.1          # 학습률
gamma = 0.95         # 할인율

Q = {}  # Q-테이블 초기화

env = SimpleTetrisEnv()

# Q-learning 알고리즘 실행
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # 상태가 처음 등장하면 Q값 0으로 초기화
        if state not in Q:
            Q[state] = np.zeros(BOARD_WIDTH)

        # 탐험 또는 최적 행동 선택
        if random.random() < epsilon:
            action = random.randint(0, BOARD_WIDTH - 1)
        else:
            action = np.argmax(Q[state])

        # 행동 수행
        next_state, reward, done = env.step(action)

        # 다음 상태 초기화
        if next_state not in Q:
            Q[next_state] = np.zeros(BOARD_WIDTH)

        # Q값 업데이트 (벨만 방정식)
        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

# Q값이 높은 상위 상태 10개 정렬
top_states = sorted(Q.items(), key=lambda x: np.max(x[1]), reverse=True)[:10]

# 데이터프레임으로 정리
df = pd.DataFrame([(s, np.round(v, 2)) for s, v in top_states], columns=["State", "Q-Values"])

# 콘솔 출력
print("\n🔝 학습된 상위 Q-값 상태들:")
print(df)

# CSV 파일로 저장
df.to_csv("top_q_values.csv", index=False)
print("\n✅ 'top_q_values.csv' 파일로 저장 완료!")
