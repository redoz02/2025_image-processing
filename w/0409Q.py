# numpy는 배열 계산, random은 행동 선택 시 사용
import numpy as np
import random

# 그리드 환경 크기 설정 (5x5)
GRID_SIZE = 5

# 가능한 행동 정의: 위(U), 아래(D), 왼쪽(L), 오른쪽(R)
ACTIONS = ['U', 'D', 'L', 'R']

# Q-Learning 하이퍼파라미터 설정
alpha = 0.1        # 학습률 (Q 값 업데이트 비율)
gamma = 0.9        # 할인율 (미래 보상 중요도)
epsilon = 0.1      # 탐험률 (랜덤 행동 확률)
episodes = 5000    # 에피소드 수 (학습 반복 횟수)

# 시작 상태 (에이전트 초기 위치)
start_state = (4, 0)

# 보상 정의: 특정 위치에서 얻는 보상값
rewards = {
    (0, 4): 1,      # 사과 위치: 보상 +1
    (0, 3): -1,     # 폭탄 위치: 보상 -1
    (4, 3): -1      # 폭탄 위치: 보상 -1
}

# 장애물 위치 정의: 이동 불가능한 셀
walls = [(2, 1), (2, 2), (2, 3)]

# Q 테이블 초기화: 각 상태마다 행동별 초기 Q값을 0으로 설정
Q = {}
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if (i, j) not in walls:
            # 해당 위치가 벽이 아닌 경우, 모든 행동에 대해 Q값 0으로 초기화
            Q[(i, j)] = {a: 0.0 for a in ACTIONS}

# 주어진 상태(state)에서 행동(action)을 했을 때의 다음 상태를 반환하는 함수
def next_state(state, action):
    i, j = state
    # 방향에 따라 상태 이동
    if action == 'U':
        i = max(i - 1, 0)  # 위로 이동 (0보다 작아지지 않도록)
    elif action == 'D':
        i = min(i + 1, GRID_SIZE - 1)  # 아래로 이동
    elif action == 'L':
        j = max(j - 1, 0)  # 왼쪽으로 이동
    elif action == 'R':
        j = min(j + 1, GRID_SIZE - 1)  # 오른쪽으로 이동

    # 이동한 위치가 벽(wall)이면 이동하지 않고 원래 위치로 유지
    if (i, j) in walls:
        return state
    return (i, j)

# 학습 루프: 여러 에피소드를 반복하여 Q값을 학습
for episode in range(episodes):
    state = start_state  # 에이전트 시작 위치 초기화

    # 보상이 있는 상태에 도달할 때까지 반복
    while state not in rewards:
        # ε-greedy 전략으로 행동 선택
        if random.uniform(0, 1) < epsilon:
            # epsilon 확률로 랜덤 행동 (탐험)
            action = random.choice(ACTIONS)
        else:
            # 나머지 경우에는 현재 Q값이 가장 높은 행동 선택 (이용)
            action = max(Q[state], key=Q[state].get)

        # 선택한 행동에 따라 다음 상태 계산
        new_state = next_state(state, action)

        # 새로운 상태에서의 보상 확인 (없으면 0)
        reward = rewards.get(new_state, 0)

        # 다음 상태에서의 최적 행동의 Q값 계산
        if new_state in Q:
            best_next_action = max(Q[new_state], key=Q[new_state].get)
            best_next_q = Q[new_state][best_next_action]
        else:
            best_next_q = 0  # 보상 상태는 행동 정의 안 돼 있음

        # Q-learning 업데이트 식 적용
        td_target = reward + gamma * best_next_q
        Q[state][action] += alpha * (td_target - Q[state][action])

        # 현재 상태를 다음 상태로 갱신
        state = new_state

# 최종적으로 상태마다 가장 좋은 행동과 가치(value)를 추출
policy = {}          # 최적 정책 저장용
value_function = {}  # 상태 가치 저장용

# Q 테이블을 기반으로 policy와 value function 구성
for state in Q:
    # 각 상태에서 가장 Q값이 높은 행동이 최적 정책
    best_action = max(Q[state], key=Q[state].get)
    best_value = Q[state][best_action]
    policy[state] = best_action
    value_function[state] = best_value

# value function을 그리드 형태로 출력
print("Value Function:")
for i in range(GRID_SIZE):
    row = ""
    for j in range(GRID_SIZE):
        if (i, j) in walls:
            row += " WALL  "       # 벽 위치 출력
        elif (i, j) in value_function:
            row += f"{value_function[(i,j)]:6.2f} "  # 값 출력
        else:
            row += "       "       # 비어 있는 공간
    print(row)

# policy를 그리드 형태로 출력
print("\nPolicy:")
for i in range(GRID_SIZE):
    row = ""
    for j in range(GRID_SIZE):
        if (i, j) in walls:
            row += "  X  "           # 벽은 X로 표시
        elif (i, j) in policy:
            row += f"  {policy[(i,j)]}  "  # 행동 표시 (U, D, L, R)
        else:
            row += "     "           # 빈 칸
    print(row)
