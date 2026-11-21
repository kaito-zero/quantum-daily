"""
qubit_sim.py: NumPy만으로 만든 1큐비트 시뮬레이터 (Day 6 미니 프로젝트)

Qiskit 없이 NumPy만 써서, 큐비트 하나에 게이트를 걸고 측정하는 일을 직접 구현한다.
Day 3(상태 벡터), Day 4(보른 규칙, 측정), Day 5(게이트 = 유니터리 행렬)에서 배운 것을
코드 하나로 묶은 것이다.

사용 예
    sim = OneQubitSim(seed=7)          # |0>에서 시작. seed를 주면 랜덤 결과가 매번 같다
    sim.run(["h", "t", "h"])           # 게이트를 차례로 건다
    print(sim.probabilities())         # [0이 나올 확률, 1이 나올 확률]
    print(sim.sample(1000))            # 1,000번 측정한 결과. 예: {'0': 853, '1': 147}
"""

import numpy as np

# ---------------------------------------------------------------
# 게이트 표: 이름 -> 2x2 유니터리 행렬
# 이름은 Qiskit의 메서드 이름(qc.h, qc.sdg ...)과 똑같이 맞췄다. 그래야 테스트에서 비교하기 쉽다.
# ---------------------------------------------------------------
_R = 1 / np.sqrt(2)                       # 1/√2 = 0.707

GATES = {
    "x": np.array([[0, 1],
                   [1, 0]], dtype=complex),            # 0과 1을 맞바꾼다 (비트 뒤집기)
    "y": np.array([[0, -1j],
                   [1j, 0]], dtype=complex),           # 맞바꾸면서 j를 곱한다 (비트 + 위상 뒤집기)
    "z": np.array([[1, 0],
                   [0, -1]], dtype=complex),           # 둘째 진폭의 부호를 뒤집는다 (위상 뒤집기)
    "h": _R * np.array([[1, 1],
                        [1, -1]], dtype=complex),      # 중첩을 만든다
    "s": np.array([[1, 0],
                   [0, 1j]], dtype=complex),           # 둘째 진폭을 90도 돌린다
    "t": np.array([[1, 0],
                   [0, np.exp(1j * np.pi / 4)]]),      # 둘째 진폭을 45도 돌린다
}
GATES["sdg"] = GATES["s"].conj().T        # S† : S를 되돌린다 (-90도)
GATES["tdg"] = GATES["t"].conj().T        # T† : T를 되돌린다 (-45도)


class OneQubitSim:
    """큐비트 하나짜리 시뮬레이터. 상태는 복소수 두 개짜리 벡터 [a, b] 하나로 들고 있는다."""

    def __init__(self, seed=None):
        self.state = np.array([1, 0], dtype=complex)    # 항상 |0>에서 시작한다
        self.rng = np.random.default_rng(seed)          # 랜덤 뽑기 기계. seed가 같으면 결과도 같다

    def apply(self, name):
        """게이트 하나를 건다 = 그 행렬을 상태 벡터에 곱한다 (Day 5)."""
        key = name.lower()
        if key not in GATES:
            raise ValueError(f"모르는 게이트: {name}. 쓸 수 있는 것: {', '.join(GATES)}")
        self.state = GATES[key] @ self.state
        return self                                     # sim.apply("h").apply("t")처럼 이어 쓸 수 있게

    def run(self, names):
        """게이트 여러 개를 왼쪽부터 차례로 건다."""
        for name in names:
            self.apply(name)
        return self

    def probabilities(self):
        """보른 규칙 (Day 4): 진폭 크기의 제곱이 확률이다."""
        return np.abs(self.state) ** 2

    def measure(self):
        """한 번 측정한다. 결과(0 또는 1)를 돌려주고, 상태는 그 결과 쪽으로 바뀐다(붕괴)."""
        p = self.probabilities()
        outcome = int(self.rng.choice(2, p=p / p.sum()))    # p.sum()으로 나누는 건 아주 작은 계산 오차 대비
        self.state = np.array([1, 0] if outcome == 0 else [0, 1], dtype=complex)
        return outcome

    def sample(self, shots):
        """같은 상태를 shots번 새로 준비해서 한 번씩 측정한 결과. 상태는 바꾸지 않는다.

        Qiskit의 get_counts()처럼 {'0': 횟수, '1': 횟수} 모양으로 돌려준다.
        """
        p = self.probabilities()
        ones = int(self.rng.binomial(shots, p[1] / p.sum()))  # shots번 중 1이 나온 횟수를 한 번에 뽑는다
        return {"0": shots - ones, "1": ones}
