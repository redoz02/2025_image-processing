from collections import defaultdict


def policy_eval(env, policy, gamma=1.0, threshold=0.0001):
    """
    정책 평가 함수.
    주어진 정책(policy)에 따라 각 상태의 가치를 계산.
    """
    V = defaultdict(float)  # 상태 가치 초기화 (모든 상태의 값은 0으로 시작)

    while True:
        delta = 0  # 가치 함수의 변화량 추적

        for state in policy.keys():
            v = V[state]
            new_v = sum([prob * (reward + gamma * V[next_state])
                         for prob, next_state, reward in policy[state]])
            V[state] = new_v

            delta = max(delta, abs(v - new_v))

        if delta < threshold:
            break

    return V
